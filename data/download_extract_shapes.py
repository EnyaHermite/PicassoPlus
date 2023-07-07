import requests
import shutil
import os
import subprocess

def download_and_extract(url):
    response = requests.get(url)
    if response.status_code == 200:
        command = f"wget {url}"
        subprocess.run(command, shell=True)

        file_name = url.split("/")[-1]
        file_path = os.path.join(os.getcwd(), file_name)
        shutil.unpack_archive(file_path, extract_dir=os.getcwd())
        os.remove(file_path)

        print("File downloaded and extracted successfully!")
    else:
        print("Failed to download the file.")


if __name__=='__main__':
    # Usage example
    meshcnn_data_url = ["https://www.dropbox.com/s/34vy4o5fthhz77d/coseg.tar.gz",
                        "https://www.dropbox.com/s/s3n05sw0zg27fz3/human_seg.tar.gz",
                        "https://www.dropbox.com/s/2bxs5f9g60wa0wr/cubes.tar.gz",
                        "https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz"]

    for url in meshcnn_data_url:
        download_and_extract(url)