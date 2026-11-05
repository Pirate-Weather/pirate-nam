"""Create the small WMS server. It only reads the public bucket, so no bucket or cloud credentials go on it. Usage: create_wms.py NAME FLAVOR REGION"""
import os, sys, time
import ovh
PROJECT, CONFIG, KEY = os.environ["OVH_PROJECT"], os.environ["OVH_CONFIG"], os.environ["NAM_SSH_KEY"]
IMAGE = "Ubuntu 24.04"
name, flavor, region = sys.argv[1:4]
c = ovh.Client(config_file=CONFIG, timeout=(10, 60))
if [i for i in c.get(f"/cloud/project/{PROJECT}/instance") if i["name"] == name]: raise SystemExit(f"{name} exists")
fl = [f["id"] for f in c.get(f"/cloud/project/{PROJECT}/flavor", region=region) if f["name"] == flavor and f["osType"] == "linux"]
img = [x["id"] for x in c.get(f"/cloud/project/{PROJECT}/image", region=region, osType="linux", flavorType=flavor) if x["name"] == IMAGE]
if not (fl and img): raise SystemExit(f"{flavor} or {IMAGE} not available in {region}")
ud = f"#cloud-config\nssh_authorized_keys:\n  - {open(KEY + '.pub').read().strip()}\npackage_update: true\npackages: [python3-venv, nginx, certbot, python3-certbot-nginx, ufw]\n"
r = c.post(f"/cloud/project/{PROJECT}/instance", flavorId=fl[0], imageId=img[0], name=name, region=region, userData=ud, monthlyBilling=False)
for _ in range(40):
    time.sleep(15); s = c.get(f"/cloud/project/{PROJECT}/instance/{r['id']}")
    if s["status"] != "BUILD": break
print(name, flavor, region, r["id"], s["status"], [a["ip"] for a in s.get("ipAddresses", []) if a["version"] == 4], flush=True)
if s["status"] != "ACTIVE": raise SystemExit("not active")
