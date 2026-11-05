"""The awphys record table parses, and every row becomes a well-formed writer field (CPU only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from namgpu.products import spec  # noqa: E402


def test_table_has_every_nam_record_in_order():
    rows = spec.read()
    assert len(rows) == 454                                   # NOAA awphys06: 404 messages, 454 fields
    assert rows[0]["name"] == "PRMSL" and rows[0]["rec"] == "1"
    recs = [r["rec"] for r in rows]
    assert len(set(recs)) == len(recs)


def test_rows_with_a_source_become_writer_fields():
    doc, left = spec.profile()
    assert doc["resolution_flags"] == 0x38
    assert len(doc["fields"]) + len(left) == 454
    for f in doc["fields"]:
        assert f["shortname"] not in ("", "-")
        assert f["template"] == (8 if f["stat"] is not None else 0)
        assert len(f["levels"]) == 1
        assert len(f["levels2"]) == 1                          # no second surface: type 255, value 0 (NCEP's bytes)
        # every record is packed at NOAA's own precision (awphys-noaa.tsv), never the flat --bits
        assert isinstance(f["decimal_scale"], int) and isinstance(f["binary_scale"], int)


def test_noaa_facts_match_the_table_row_by_row():
    rows = spec.attach_noaa(spec.read())
    assert all("noaa_D" in r and r["noaa_name"] and r["noaa_level"] for r in rows)
    by = {r["rec"]: r for r in rows}
    assert (by["1"]["noaa_name"], by["1"]["noaa_level"], by["1"]["noaa_D"], by["1"]["noaa_E"]) == \
        ("PRMSL", "mean sea level", 2, 4)
    assert by["3"]["noaa_level"] == "entire atmosphere (considered as a single layer)"   # wgrib2's text for type 200
    assert by["293"]["noaa_name"] == "MSLET"
    tsoil = next(r for r in rows if r["name"] == "TSOIL" and r["level2"] != "-")
    assert tsoil["noaa_S1"] == 2 and tsoil["noaa_S2"] == 2     # NOAA writes 0-0.1 m as 0 and 10 x 10^-2


def test_core_records_are_mapped():
    by = {(r["name"], r["level"]): r for r in spec.read()}
    assert by[("TMP", "103:2")]["source"] == "TMP_ON_SPEC_HGT_LVL_ABOVE_GRND_2m"
    assert by[("MSLET", "101:0")]["source"] == "MSLET_ON_MEAN_SEA_LVL"
    assert by[("HGT", "100:50000")]["source"] == "HGT_ON_ISOBARIC_SFC"


def test_isobaric_group_is_mapped():
    # every isobaric record of awphys06 (HGT/TMP/RH/UGRD/VGRD/VVEL/TKE x39, DZDT x42, ABSV x5) has a source, and the
    # profile asks the writer for the NMM isobaric rule (NAM post MDL2P), not the WOOF post isobaric fields
    rows = [r for r in spec.read() if r["t1"] == 100 and r["t2"] is None]
    count = {}
    for r in rows:
        count[r["name"]] = count.get(r["name"], 0) + 1
        assert r["source"] not in ("", "-"), (r["rec"], r["name"], r["level"])
    assert count == {"HGT": 39, "TMP": 39, "RH": 39, "UGRD": 39, "VGRD": 39, "VVEL": 39, "DZDT": 42, "ABSV": 5,
                     "var0_19_11": 39}
    assert {r["source"] for r in rows if r["name"] == "var0_19_11"} == {"Q2_ON_ISOBARIC_SFC"}
    assert spec.profile()[0]["isobaric_rule"] == "upp-nmm"


def test_convective_and_reflectivity_records_are_mapped():
    """Convective, precipitation and reflectivity: NAM's own post definitions (namgpu/products/awphys.tsv notes; gpuwm rrfs_upp.rs module namc)."""
    by = {r["rec"]: r for r in spec.read()}
    want = {
        "2": "REFD_ON_HYBRID_LVL", "3": "carrier:COMPOSITE", "11": "NAM_RETOP_ON_ENTIRE_ATMOS",
        "278": "NAM_HINDEX_ON_SURFACE", "295": "NAM_REFD_ON_SPEC_HGT_LVL_ABOVE_GRND",
        "296": "NAM_REFD_ON_SPEC_HGT_LVL_ABOVE_GRND", "349": "NAM_LFTX_ON_ISOBARIC_SFC_500-1000hpa",
        "350": "NAM_CAPE_ON_SURFACE", "351": "NAM_CIN_ON_SURFACE", "352": "PWAT_ON_ENTIRE_ATMOS_SINGLE_LYR",
        "354": "NAM_HLCY_ON_SPEC_HGT_LVL_ABOVE_GRND", "355": "NAM_VUCSH_ON_SPEC_HGT_LVL_ABOVE_GRND_0-6km",
        "356": "NAM_VVCSH_ON_SPEC_HGT_LVL_ABOVE_GRND_0-6km", "371": "NAM_PLI_ON_SPEC_PRES_ABOVE_GRND_30-0mb",
        "387": "NAM_4LFTX_ON_SPEC_PRES_ABOVE_GRND_180-0mb", "388": "NAM_CAPE_ON_SPEC_PRES_ABOVE_GRND_180-0mb",
        "389": "NAM_CIN_ON_SPEC_PRES_ABOVE_GRND_180-0mb",
    }
    for rec, source in want.items():
        assert by[rec]["source"] == source, (rec, by[rec]["source"])
    # 0.2.15/0.2.16 at 0-6000 m in NAM's awphys are the vertical shear (VUCSH/VVCSH), not storm motion
    assert (by["355"]["dcp"], by["355"]["level"], by["355"]["level2"]) == ("0.2.15", "103:0", "103:6000")
    # records the port cannot produce honestly stay unmapped, with the reason in the note
    for rec in ("334", "336", "337", "338", "339", "392"):
        assert by[rec]["source"] in ("", "-") and by[rec]["note"].startswith("LEFT OUT"), rec


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)


def test_group_a_records_are_mapped():
    """Surface and near-surface: every record with an honest source is mapped; the two NAM records with
    no method to port (visibility at cloud top, the all-zero Blackadar length) stay listed as left out."""
    rows = {r["rec"]: r for r in spec.read()}
    mapped = ["1", "4", "7", "8.1", "8.2", "9", "10", "297", "298", "299", "322", "323", "324", "325", "326", "327",
              "328", "329.1", "329.2", "330", "331", "335", "357", "358", "359.1", "359.2", "360", "361.1", "361.2",
              "362", "363", "364", "365.1", "365.2", "366", "367", "390", "391"]
    for base in (368, 372, 375, 378, 381, 384):
        mapped += [str(base), str(base + 1), f"{base + 2}.1", f"{base + 2}.2"]
    for rec in mapped:
        assert rows[rec]["source"] not in ("", "-"), rec
    for rec in ("5", "6"):
        assert rows[rec]["source"] == "-" and rows[rec]["note"].startswith("left out"), rec
    # the 30 hPa layers carry both surfaces; the sigma layer is 0.33-1.0
    assert (rows["384"]["t1"], rows["384"]["v1"], rows["384"]["t2"], rows["384"]["v2"]) == (108, 18000.0, 108, 15000.0)
    assert (rows["391"]["v1"], rows["391"]["v2"]) == (0.33, 1.0)
