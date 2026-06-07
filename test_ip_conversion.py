import sys
sys.path.append('src')

from geolocation import ip_to_int

# Test IPs
test_cases = [
    ('192.168.1.1', 3232235521),
    ('0.0.0.0', 0),
    ('255.255.255.255', 4294967295),
    ('10.0.0.1', 167772161),
    ('8.8.8.8', 134744072),
]

print("Testing IP conversion:")
for ip, expected in test_cases:
    result = ip_to_int(ip)
    status = "✓" if result == expected else "✗"
    print(f"{status} {ip} -> {result} (expected: {expected})")
