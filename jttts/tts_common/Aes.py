from Crypto.Cipher import AES # pip install pycryptodome
import base64
import sys
import time

class Aescrypt():
    def __init__(self,key,model,iv):
        self.key = self.add_16(key)
        self.model = model
        self.iv = iv

    def add_16(self,par):
        if type(par) == str:
            par = par.encode()
        while len(par) % 16 != 0:
            par += b'\x00'
        return par

    def aesencrypt(self,text):
        text = self.add_16(text)
        if self.model == AES.MODE_CBC:
            self.aes = AES.new(self.key,self.model,self.iv.encode('utf-8')) 
        elif self.model == AES.MODE_ECB:
            self.aes = AES.new(self.key,self.model) 
        self.encrypt_text = self.aes.encrypt(text)
        return self.encrypt_text

    def aesdecrypt(self,text):
        text = self.add_16(text)
        if self.model == AES.MODE_CBC:
            self.aes = AES.new(self.key,self.model,self.iv.encode('utf-8')) 
        elif self.model == AES.MODE_ECB:
            self.aes = AES.new(self.key,self.model) 
        self.decrypt_text = self.aes.decrypt(text)
        self.decrypt_text = self.decrypt_text.strip(b"\x00")
        return self.decrypt_text



