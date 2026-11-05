"""Create one named GPU test instance. Usage: create_gpu2.py NAME FLAVOR REGION [REGION...]. No credentials go on it: third-party code runs there."""
import os
import ovh, time, sys
c = ovh.Client(config_file=os.environ["OVH_CONFIG"], timeout=(10, 60)); pid = os.environ["OVH_PROJECT"]; name, flavor = sys.argv[1:3]
if [i for i in c.get(f"/cloud/project/{pid}/instance") if i["name"] == name]: raise SystemExit(f"{name} exists")
pub = open(os.environ["NAM_SSH_KEY"] + ".pub").read().strip()
ud = f"#cloud-config\nssh_authorized_keys:\n  - {pub}\npackage_update: true\npackages: [rsync, python3-venv, docker.io]\n"
for region in sys.argv[3:]:
    fl = [f["id"] for f in c.get(f"/cloud/project/{pid}/flavor", region=region) if f["name"] == flavor and f["osType"] == "linux"]
    img = [x for x in c.get(f"/cloud/project/{pid}/image", region=region, osType="linux", flavorType=flavor) if x["name"] == "Ubuntu 24.04 - NVIDIA - v580"]
    if not (fl and img): print(name, region, "no flavour or image", flush=True); continue
    keys = [k for k in c.get(f"/cloud/project/{pid}/sshkey", region=region) if k["name"] == os.environ.get("OVH_SSH_KEY_NAME")]
    try:
        r = c.post(f"/cloud/project/{pid}/instance", flavorId=fl[0], imageId=img[0]["id"], name=name, region=region, sshKeyId=keys[0]["id"] if keys else None, userData=ud, monthlyBilling=False)
    except ovh.exceptions.APIError as e:
        print(name, region, "refused:", str(e)[:160], flush=True); continue
    for _ in range(40):
        time.sleep(15); s = c.get(f"/cloud/project/{pid}/instance/{r['id']}")
        if s["status"] != "BUILD": break
    print(name, flavor, region, r["id"], s["status"], [a["ip"] for a in s.get("ipAddresses", []) if a["version"] == 4], flush=True)
    if s["status"] == "ACTIVE": raise SystemExit
    c.delete(f"/cloud/project/{pid}/instance/{r['id']}"); time.sleep(20)
print(name, "NONE"); sys.exit(1)
