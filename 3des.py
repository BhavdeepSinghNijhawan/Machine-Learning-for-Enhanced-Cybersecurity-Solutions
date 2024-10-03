from Crypto.Cipher import DES
from Crypto.Hash import SHA256
from getpass import getpass
from Crypto.Protocol.KDF import PBKDF2

pi = 100005
salt_const = b"$ez*}-d3](%d%$#*!)$#%s45le$*fhucdivyanshu75456dgfdrrrrfgfs^"

def encryptor(path):
    try:
        with open(path, 'rb') as imagefile:
            image = imagefile.read()
        while len(image) % 8 != 0:
            image += b" "
    except:
        print("Error loading the file, make sure file is in same directory, spelled correctly and non-corrupted")
        exit()

    hash_of_original = SHA256.new(data=image)

    key_enc = getpass(prompt="Enter minimum 8 character long password:")
    while len(key_enc) < 8:
        key_enc = getpass(prompt="Invalid password! Enter atleast 8 character password:")

    key_enc_confirm = getpass(prompt="Enter password again:")
    while key_enc != key_enc_confirm:
        print("Key Mismatch. Try again.")
        key_enc = getpass(prompt="Enter 8 character long password:")
        while len(key_enc) < 8:
            key_enc = getpass(prompt="Invalid password! Enter atleast 8 character password:")
        key_enc_confirm = getpass(prompt="Enter password again:")

    key_enc = PBKDF2(key_enc, salt_const, 48, count=pi)

    print("Encrypting...")

    try:
        cipher1 = DES.new(key_enc[0:8], DES.MODE_CBC, key_enc[24:32])
        ciphertext1 = cipher1.encrypt(image)
        cipher2 = DES.new(key_enc[8:16], DES.MODE_CBC, key_enc[32:40])
        ciphertext2 = cipher2.decrypt(ciphertext1)
        cipher3 = DES.new(key_enc[16:24], DES.MODE_CBC, key_enc[40:48])
        ciphertext3 = cipher3.encrypt(ciphertext2)
        print("Encryption successful!")
    except:
        print("Encryption failed... Possible causes: Library not installed properly/low device memory/Incorrect padding or conversions")
        exit()

    ciphertext3 += hash_of_original.digest()

    try:
        dpath = "encrypted_" + path
        with open(dpath, 'wb') as image_file:
            image_file.write(ciphertext3)
        print(f"Encrypted Image Saved successfully as filename {dpath}")
    except:
        temp_path = input("Saving file failed! Enter alternate name without format to save the encrypted file.")
        try:
            dpath = "encrypted_" + path
            with open(dpath, 'wb') as image_file:
                image_file.write(ciphertext3)
            print(f"Encrypted Image Saved successfully as filename {dpath}")
            exit()
        except:
            print("Failed...Exiting...")
            exit()

def decryptor(encrypted_image_path):
    try:
        with open(encrypted_image_path, 'rb') as encrypted_file:
            encrypted_data_with_hash = encrypted_file.read()
    except:
        print("Unable to read source cipher data. Make sure the file is in same directory...Exiting...")
        exit()

    key_dec = getpass(prompt="Enter password:")

    extracted_hash = encrypted_data_with_hash[-32:]
    encrypted_data = encrypted_data_with_hash[:-32]

    key_dec = PBKDF2(key_dec, salt_const, 48, count=pi)

    print("Decrypting...")

    try:
        cipher1 = DES.new(key_dec[16:24], DES.MODE_CBC, key_dec[40:48])
        plaintext1 = cipher1.decrypt(encrypted_data)
        cipher2 = DES.new(key_dec[8:16], DES.MODE_CBC, key_dec[32:40])
        plaintext2 = cipher2.encrypt(plaintext1)
        cipher3 = DES.new(key_dec[0:8], DES.MODE_CBC, key_dec[24:32])
        plaintext3 = cipher3.decrypt(plaintext2)
    except:
        print("Decryption failed... Possible causes: Library not installed properly/low device memory/Incorrect padding or conversions")

    hash_of_decrypted = SHA256.new(data=plaintext3)

    if hash_of_decrypted.digest() == extracted_hash:
        print("Password Correct !!!")
        print("Decryption successful!")
    else:
        print("Incorrect Password!!!")
        exit()

    try:
        epath = encrypted_image_path
        if epath[:10] == "encrypted_":
            epath = epath[10:]
        epath = "decrypted_" + epath
        with open(epath, 'wb') as image_file:
            image_file.write(plaintext3)
        print(f"Image saved successfully with name {epath}")
    except:
        temp_path = input("Saving file failed! Enter alternate name without format to save the decrypted file.")
        try:
            epath = temp_path + encrypted_image_path
            with open(epath, 'wb') as image_file:
                image_file.write(plaintext3)
            print(f"Image saved successfully with name {epath}")
        except:
            print("Failed! Exiting...")
            exit()

try:
    choice = int(input("Press 1 for Encryption || 2 for Decryption: "))
    while choice != 1 and choice != 2:
        choice = int(input("Invalid Choice! Try Again:"))
except:
    print("Error, please provide valid Input")
    exit()

if choice == 1:
    path = input("Enter image's name to be encrypted:")
    encryptor(path)
else:
    encrypted_image_path = input("Enter file name to decrypt:")
    decryptor(encrypted_image_path)
