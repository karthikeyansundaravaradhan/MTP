import zipfile, os
from pathlib import Path

DD = Path("/Users/karthikeyansundaravaradhan/Desktop/Masters Project/DriverDrowsinessDataset")

ZIPS = {
    "CSV_Files.zip": DD / "CSV_Files",
    "Codes.zip": DD / "Codes",
    "Extracted_Features.zip": DD / "Extracted_Features",
    "Video_Data.zip": DD / "Video_Data",
}

overall_ok = True
for zip_name, extract_root in ZIPS.items():
    zip_path = DD / zip_name
    print(f"\n=== {zip_name} -> {extract_root} ===")
    if not zip_path.exists():
        print("  ZIP MISSING")
        overall_ok = False
        continue

    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]

    n_total = len(infos)
    n_ok = 0
    n_missing = 0
    n_size_mismatch = 0
    missing_examples = []
    mismatch_examples = []

    for info in infos:
        # zip entries are relative to the zip's own root; the extracted layout
        # mirrors that under extract_root's parent (DD), since these were
        # originally unzipped directly into DD.
        disk_path = DD / info.filename
        if not disk_path.exists():
            n_missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(info.filename)
            continue
        actual_size = disk_path.stat().st_size
        if actual_size != info.file_size:
            n_size_mismatch += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append((info.filename, info.file_size, actual_size))
            continue
        n_ok += 1

    print(f"  Files in zip     : {n_total}")
    print(f"  Present + size OK: {n_ok}")
    print(f"  Missing on disk  : {n_missing}")
    print(f"  Size mismatches  : {n_size_mismatch}")
    if missing_examples:
        print(f"  Missing examples : {missing_examples}")
    if mismatch_examples:
        print(f"  Mismatch examples: {mismatch_examples}")

    if n_missing > 0 or n_size_mismatch > 0:
        overall_ok = False
        print("  -> NOT SAFE TO DELETE")
    else:
        print("  -> OK, safe to delete this zip")

print("\n" + "=" * 50)
print("ALL ZIPS VERIFIED SAFE TO DELETE" if overall_ok else "DO NOT DELETE — mismatches found")
