"""Generate a two-row timeline of the graded public window and prediction."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'H':'#2469ac', 'A':'#b43852', 'D':'#e8a82f', 'X':'#bfc6ce'}


def read_labels(path):
    with path.open(encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ['frame','label']:
            raise ValueError(f'Expected frame,label header in {path}')
        labels = []
        for index, row in enumerate(reader):
            if int(row['frame']) != index or row['label'] not in COLORS:
                raise ValueError(f'Invalid row {index} in {path}')
            labels.append(row['label'])
    if len(labels) != 199319:
        raise ValueError(f'Unexpected video length in {path}')
    return labels


def runs(values):
    start = 0
    for stop in range(1, len(values)+1):
        if stop == len(values) or values[start] != values[stop]:
            yield start, stop, values[start]
            start = stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-gt', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, default=ROOT/'evaluation/results/AnkaraMessi.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/results/img/public_window.svg')
    args = parser.parse_args()
    if args.public_gt.name != 'BAR-ATH_possession_gt_public.csv':
        raise ValueError('Use the official public GT filename')
    gt, prediction = map(read_labels, (args.public_gt, args.prediction))
    spans = [(a,b) for a,b,graded in runs([label!='X' for label in gt]) if graded]
    if sum(b-a for a,b in spans) != 7500:
        raise ValueError('Expected exactly 7500 public graded frames')
    width, left, plot_width = 1320, 170, 1120
    height = 110 + 125*len(spans)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           'role="img" aria-labelledby="title description">',
           '<title id="title">BAR–ATH · finestra pubblica</title>',
           '<desc id="description">Due righe per intervallo: riferimento pubblico e predizione AnkaraMessi.</desc>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="system-ui,sans-serif" fill="#202b38">',
           '<text x="24" y="33" font-size="24" font-weight="700">BAR–ATH · GT pubblico e AnkaraMessi</text>',
           '<text x="24" y="57" font-size="13">7.500 frame valutabili · tempo in secondi dal primo frame della finestra</text>']
    for i,(a,b) in enumerate(spans):
        y = 75 + 125*i
        duration = (b-a)/25
        for row,(name,values) in enumerate((('GT pubblico',gt), ('AnkaraMessi',prediction))):
            top = y+row*34
            svg.append(f'<text x="24" y="{top+17}" font-size="14">{name}</text>')
            for start,stop,label in runs(values[a:b]):
                x = left + plot_width*start/(b-a)
                w = plot_width*(stop-start)/(b-a)
                svg.append(f'<rect x="{x:.4f}" y="{top}" width="{w:.4f}" height="24" fill="{COLORS[label]}">'
                           f'<title>{label} · frame {a+start}–{a+stop-1}</title></rect>')
        for tick in range(6):
            x = left+plot_width*tick/5
            svg.append(f'<text x="{x:.2f}" y="{y+82}" font-size="12" text-anchor="middle">{duration*tick/5:g}</text>')
    for i,(key,name) in enumerate(zip('HADX',('H · Barcelona','A · Athletic Club','D · palla morta','X · astensione'))):
        x = 24+i*235
        svg.append(f'<rect x="{x}" y="{height-30}" width="16" height="16" fill="{COLORS[key]}"/>'
                   f'<text x="{x+23}" y="{height-17}" font-size="13">{name}</text>')
    svg.append('</g></svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(svg)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
