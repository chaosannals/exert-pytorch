import os
import asyncio
from aiofile import AIOFile
from aiohttp import ClientSession, ClientTimeout
from urllib.request import urlopen

model_file_links = [
    'https://cdn.huggingface.co/Helsinki-NLP/opus-mt-zh-en/README.md',
    'https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP/opus-mt-zh-en/config.json',
    'https://cdn.huggingface.co/Helsinki-NLP/opus-mt-zh-en/metadata.json',
    'https://cdn.huggingface.co/Helsinki-NLP/opus-mt-zh-en/pytorch_model.bin',
    'https://cdn.huggingface.co/Helsinki-NLP/opus-mt-zh-en/source.spm',
    'https://cdn.huggingface.co/Helsinki-NLP/opus-mt-zh-en/target.spm',
    'https://cdn.huggingface.co/Helsinki-NLP/opus-mt-zh-en/tokenizer_config.json',
    'https://cdn.huggingface.co/Helsinki-NLP/opus-mt-zh-en/vocab.json',
]

async def download(url, folder, file_name):
    headers = {
        'User-agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5'
    }
    timeout = ClientTimeout(total=86400)
    async with ClientSession(headers=headers, timeout=timeout) as session:
        async with session.get(url) as r:
            target = os.path.join(folder, file_name)
            async with AIOFile(target, 'wb') as afp:
                await afp.write(await r.read())
                await afp.fsync()

async def download_to(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    for link in model_file_links:
        print(f'download: {link}')
        file_name = os.path.basename(link)
        await download(link, folder, file_name)

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_to('./hfmt/model/zh2en'))

if __name__ == '__main__':
    main()