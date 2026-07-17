import logging
import os
import time

import base64

encryption_key      = "jttts_qrewyue_#4^@%*&"
encryption_label    = "enc_xy@#!_content:"

def xor_encrypt(text, key):
    """
    对输入文字进行 XOR 加密
    :param text: 待加密的文字
    :param key: 加密密钥
    :return: 加密后的字节数据
    """
    text_bytes = text.encode('utf-8')
    key_bytes = key.encode('utf-8')
    encrypted = bytearray()
    key_length = len(key_bytes)
    for i, byte in enumerate(text_bytes):
        encrypted.append(byte ^ key_bytes[i % key_length])
    return bytes(encrypted)


def encrypt_and_encode(text, key = encryption_key):
    """
    先对文字进行 XOR 加密，再进行 Base64 编码
    :param text: 待加密的文字
    :param key: 加密密钥
    :return: Base64 编码后的加密字符串
    """
    encrypted_bytes = xor_encrypt(text, key)
    encoded = base64.b64encode(encrypted_bytes).decode('utf-8')
    return encoded


def decode_and_decrypt(encoded_text, key = encryption_key):
    """
    先对 Base64 编码的字符串进行解码，再进行 XOR 解密
    :param encoded_text: Base64 编码后的加密字符串
    :param key: 解密密钥（与加密密钥相同）
    :return: 解密后的原始文字
    """
    encrypted_bytes = base64.b64decode(encoded_text)
    key_bytes = key.encode('utf-8')
    decrypted_bytes = bytearray()
    key_length = len(key_bytes)
    for i, byte in enumerate(encrypted_bytes):
        decrypted_bytes.append(byte ^ key_bytes[i % key_length])
    return decrypted_bytes.decode('utf-8')


if __name__ == '__main__':

    # 示例使用
    original_text = "Hello,7892, 世界! 这是一个测试。\n 今天天气不错*7^^$##@!"
    encryption_key = "testkfdsdfa5894743#@4ey"

    # 加密并编码
    encrypted_encoded = encrypt_and_encode(original_text, encryption_key)
    print(f"加密并 Base64 编码后的字符串: {encrypted_encoded}")

    # 解码并解密
    decrypted_text = decode_and_decrypt(encrypted_encoded, encryption_key)
    print(f"解密后的原始文本: {decrypted_text}")

    if original_text == decrypted_text:
        print(f'test success')