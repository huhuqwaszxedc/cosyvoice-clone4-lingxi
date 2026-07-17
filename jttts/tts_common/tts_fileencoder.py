# coding=utf-8

# **********************************************************************
from jttts.tts_common.logger import logger
import jttts.tts_model.frontend.config_frontend as cfg
from jttts.tts_common.Aes     import Aescrypt
from Crypto.Cipher      import AES  # pip install pycryptodome
import io, tempfile, time, os
import uuid

passwd          = "yangwetyhjuytrfd"
iv              = "yangrtfghjkuyrtg"
aescryptor      = Aescrypt(passwd,AES.MODE_CBC,iv)


def check_encrypt_file_valid(origin_filepath, loginfo = ''):
    enc_suffix = cfg.enc_suffix

    sub_filename = origin_filepath.split('/')[-1]
    if sub_filename[-4:] == '.txt':
        new_sub_filename = sub_filename[:-4] + f'_{enc_suffix}.txt'
    else:
        new_sub_filename = sub_filename + '_' + enc_suffix

    enc_filepath = os.path.join(os.path.dirname(origin_filepath), new_sub_filename)


    if os.path.exists(enc_filepath):
        return enc_filepath, True
    elif os.path.exists(origin_filepath):
        return origin_filepath, False
    else:
        logger.info(f"{loginfo} file is not exists! please check !:{origin_filepath}  or {enc_filepath} ")
        exit()


# 加密
def text_file_encrypt(in_filepath, output_filepath):
    if os.path.exists(in_filepath) is False:
        print(f'{in_filepath} is not exists!!')
        exit()
    with open(in_filepath,'rb') as f:
        txt_binary = f.read()
    encrypt_aes  = aescryptor.aesencrypt(txt_binary)
    with open(output_filepath, 'wb') as f:
        f.write(encrypt_aes)

def write_dict_to_encryptfile(data_dict, output_filepath):
    
    output_filepath, is_encrypt = check_encrypt_file_valid(output_filepath, loginfo = 'frontrepair')
    if is_encrypt:
        os.makedirs('/tmp', exist_ok=True)
        tmp_file = os.path.join('/tmp', 'tmpf_' + str(round(time.time() * 1000))) 
        with open(tmp_file, 'w', encoding='utf-8') as fid:
            for key, value in data_dict.items():
                fid.write(f'{key}\t{value} \n')
        text_file_encrypt(tmp_file, output_filepath)
        os.remove(tmp_file)
    else:
        with open(output_filepath, 'w', encoding='utf-8') as fid:
            for key, value in data_dict.items():
                fid.write(f'{key}\t{value} \n')



# 解密

def text_file_decrypt(filepath):
    with open(filepath,'rb') as f:
        read_data_aes = f.read()
    decrypted_data = aescryptor.aesdecrypt(read_data_aes) 

    binary_io = io.BytesIO()
    binary_io.write(decrypted_data)
    binary_io.seek(0)

    # 把二进制数据解码成字符串
    decoded_text = binary_io.read().decode('utf-8')

    # 模拟 readlines 方法按行分割字符串
    decrypted_txt_list = decoded_text.splitlines(keepends=True)

    return decrypted_txt_list

def text_file_decrypt_to_tmpfile(filepath):

    with open(filepath,'rb') as f:
        read_data_aes = f.read()
    decrypted_data = aescryptor.aesdecrypt(read_data_aes) 

    binary_io = io.BytesIO()
    binary_io.write(decrypted_data)
    binary_io.seek(0)

    # 把二进制数据解码成字符串
    decoded_text = binary_io.read().decode('utf-8')

    # 模拟 readlines 方法按行分割字符串
    decrypted_txt_list = decoded_text.splitlines(keepends=True)
    current_millis = str(round(time.time() * 1000))
    temp_file_path = os.path.join('/tmp', current_millis)

    with open(temp_file_path,'w') as tempf:
        for sub_txt in decrypted_txt_list:
            tempf.write(sub_txt)

    return temp_file_path

