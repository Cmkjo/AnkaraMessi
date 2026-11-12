"""Download the DFL XML files identified by the training recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

API = 'https://api.figshare.com/v2/articles/28196177'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    recipe=json.loads(Path(__file__).with_name('recipe.json').read_text())
    expected=recipe['input_sha256']
    with urllib.request.urlopen(API) as response:
        metadata=json.load(response)
    files={row['name']:row for row in metadata['files']}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,digest in expected.items():
        path=args.output_dir/name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:
            continue
        if name not in files:
            raise ValueError(f'Missing source file: {name}')
        temp=path.with_suffix('.download')
        try:
            with urllib.request.urlopen(files[name]['download_url']) as response,temp.open('wb') as output:
                for block in iter(lambda:response.read(1024*1024),b''):
                    output.write(block)
            if hashlib.sha256(temp.read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Unexpected dataset version: {name}')
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    (args.output_dir/'figshare_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    main()
