import json, zipfile

with zipfile.ZipFile("task_sdfsdfdsf_backup_2026_11_14_16_38_38.zip") as z:
    names = z.namelist()

    print("=== files in archive (order) ===")
    for n in names:
        print(" ", n)

    print("\n=== manifest.jsonl ===")
    for line in z.read("data/manifest.jsonl").decode().splitlines():
        if line.strip():
            print(" ", line)

    print("\n=== task.json (key fields) ===")
    task = json.loads(z.read("task.json"))
    for k in ["name", "subset", "status", "version"]:
        print(f"  {k}: {task.get(k)!r}")
    print("  data.start_frame:", task["data"]["start_frame"])
    print("  data.stop_frame: ", task["data"]["stop_frame"])
    print("  data.chunk_size: ", task["data"]["chunk_size"])
    print("  data.sorting_method:", task["data"]["sorting_method"])
    print("  jobs:", task["jobs"])

    print("\n=== annotations.json (structure) ===")
    ann = json.loads(z.read("annotations.json"))
    print("  len(annotations):", len(ann))
    for i, a in enumerate(ann):
        shapes = a.get("shapes", [])
        frames = sorted({s.get("frame") for s in shapes})
        print(f"  [{i}] shapes={len(shapes)} frames={frames}")
        if shapes:
            s0 = shapes[0]
            print(f"       first shape keys: {sorted(s0.keys())}")
            print(f"       first shape points type: {type(s0['points']).__name__}, "
                  f"len={len(s0['points'])}, tail={s0['points'][-4:]}")