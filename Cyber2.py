from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
# Generate RSA key pair
key = RSA.generate(2048)
public_key = key.publickey()
encryptor = PKCS1_OAEP.new(public_key)
decryptor = PKCS1_OAEP.new(key)
# Data to encrypt
data = "Cybersecurity Lab RSA Example".encode('utf-8')
# Encrypt
encrypted = encryptor.encrypt(data)
print("Encrypted:", base64.b64encode(encrypted))
# Decrypt
decrypted = decryptor.decrypt(encrypted)
print("Decrypted:", decrypted.decode('utf-8'))