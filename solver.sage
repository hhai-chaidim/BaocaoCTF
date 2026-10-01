import json

with open('out.txt') as f:
    d = json.load(f)

p, n, m = 17, 32, 34
F = GF(p)
R = PolynomialRing(F, n, 'x', order='degrevlex')
x = R.gens()

P = []
for poly in d['public_key']['polynomials']:
    poly_dict = {}
    for c, mon in poly:
        deg = [0] * n
        for v in mon: deg[v] += 1
        poly_dict[tuple(deg)] = F(c)
    P.append(R(poly_dict))

def decode(digits):
    w = 1
    while p**w < 256: w += 1
    if len(digits) < 4*w: return b""
    
    def dec_byte(s): 
        val = 0
        for digit in digits[s:s+w]: val = val * p + digit
        return val
        
    try:
        length = int.from_bytes(bytes(dec_byte(i*w) for i in range(4)), 'big')
        end = (4 + length) * w
        return bytes(dec_byte(i) for i in range(4*w, end, w))
    except: 
        return b""

flag = b""
field_eqs = [x[i]^p - x[i] for i in range(n)]

for block in d['ciphertext']['blocks']:
    eqs = [P[i] - F(block[i]) for i in range(m)] + field_eqs
    gb = R.ideal(eqs).groebner_basis()
    
    vec = [0] * n
    for poly in gb:
        if poly.degree() == 1:
            var_idx = R(poly - poly.constant_coefficient()).lm().index()
            vec[var_idx] = int(-poly.constant_coefficient())
            
    flag += decode(vec)

print(flag.decode('utf-8', errors='ignore'))