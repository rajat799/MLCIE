from Crypto.Cipher import AES
import base64
# Padding for data
def pad(data):
 return data + (16 - len(data) % 16) * chr(16 - len(data) % 16)
def unpad(data):
 return data[:-ord(data[-1])]
# Key must be 16, 24 or 32 bytes long
key = "thisisasecretkey".encode('utf-8')
# Encrypt
data = "Cybersecurity Lab AES Example"
cipher = AES.new(key, AES.MODE_ECB)
encrypted = base64.b64encode(cipher.encrypt(pad(data).encode('utf-8')))
print("Encrypted:", encrypted)
# Decrypt
decrypted = unpad(cipher.decrypt(base64.b64decode(encrypted)).decode('utf-8'))
print("Decrypted:", decrypted)