import json

# 기존 CCVID.json 경로
with open('./datasets/CCVID/CCVID.json', 'r') as f:
    old_data = json.load(f)

# 새로운 구조 준비
new_data = {
    "TRAIN_SET": [],
    "TEST_SET": [],
    "GALLERY_SET": [],
    "PROBE_SET": []
}

# label2seq 사용해서 train/test 항목 구성
for label, seqs in old_data['label2seq'].items():
    for session, cam in seqs:
        sample_id = f"{label}_{session}_{cam}"
        if label in old_data['train']:
            new_data["TRAIN_SET"].append(sample_id)
        if label in old_data['test']['gallery']:
            new_data["GALLERY_SET"].append(sample_id)
        if label in old_data['test']['probe']:
            new_data["PROBE_SET"].append(sample_id)

# TEST_SET = GALLERY_SET + PROBE_SET (중복 제거)
new_data["TEST_SET"] = list(set(new_data["GALLERY_SET"] + new_data["PROBE_SET"]))

# 저장
with open('./datasets/CCVID/CCVID_partition.json', 'w') as f:
    json.dump(new_data, f, indent=4)

print("변환 완료: CCVID_partition.json 생성됨")
