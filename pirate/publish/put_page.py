"""Upload the viewer page and data page only. viewer/publish.py rebuilds runs.json from local folders, which would drop WOOF runs."""
import os
import ovh, boto3
c = ovh.Client(config_file=os.environ["OVH_CONFIG"], timeout=(10, 60)); pid = os.environ["OVH_PROJECT"]; uid = int(os.environ["OVH_USER"])
ak = c.get(f"/cloud/project/{pid}/user/{uid}/s3Credentials")[0]["access"]
sk = c.post(f"/cloud/project/{pid}/user/{uid}/s3Credentials/{ak}/secret")["secret"]
s3 = boto3.client("s3", endpoint_url="https://s3.ca-east-tor.io.cloud.ovh.net", region_name="ca-east-tor", aws_access_key_id=ak, aws_secret_access_key=sk)
for name, ctype in (("index.html", "text/html; charset=utf-8"), ("data.html", "text/html; charset=utf-8"), ("logo.svg", "image/svg+xml")):
    s3.upload_file(f"{os.environ['PIRATE_VIEWER']}/{name}", "pirate-nam-tor", name,
                   ExtraArgs={"ACL": "public-read", "ContentType": ctype, "CacheControl": "public, max-age=60", "StorageClass": "EXPRESS_ONEZONE"})
s3.upload_file(os.path.join(os.environ["PIRATE_VIEWER"], "geo/lines.json"), "pirate-nam-tor", "geo/lines.json",
               ExtraArgs={"ACL": "public-read", "ContentType": "application/json", "CacheControl": "public, max-age=86400", "StorageClass": "EXPRESS_ONEZONE"})
print("pages uploaded")
