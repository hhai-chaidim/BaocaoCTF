import json

# 1. Đọc dữ liệu thô
with open('out.txt', 'r') as f:
    data = json.load(f)

# 2. Khảo sát tham số và số lượng khối mã hóa
params = data['public_key']['parameters']
cipher_blocks = data['ciphertext']['blocks']
print(f"Tham số: {params}")
print(f"Số khối bản mã (blocks) cần giải quyết: {len(cipher_blocks)}")

# 3. Khảo sát bề mặt đa thức công khai đầu tiên (P_0)
polys = data['public_key']['polynomials']
print("\n[!] Cấu trúc 5 hạng tử đầu tiên của đa thức P_0:")
for term in polys[0][:5]:
    print(term)

# 4. Kiểm chứng sự vắng mặt của khóa bí mật
print("\n[?] Khóa bí mật có bị rò rỉ không?:", "private" in data)