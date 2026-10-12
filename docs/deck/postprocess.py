"""Drop empty numeric points from chart caches and show blanks as gaps (curves of different lengths)."""
import re, sys, zipfile, shutil
src, dst = sys.argv[1], sys.argv[2]
tmp = dst + '.tmp'
changed = []
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if re.match(r'ppt/charts/chart\d+\.xml$', item.filename):
            xml = data.decode('utf-8')
            new, n = re.subn(r'<c:pt idx="\d+"><c:v></c:v></c:pt>', '', xml)
            if n:
                if '<c:dispBlanksAs' in new:
                    new = re.sub(r'<c:dispBlanksAs val="\w+"/>', '<c:dispBlanksAs val="gap"/>', new)
                else:
                    new, k = re.subn(r'(<c:plotVisOnly val="[01]"/>)', r'\1<c:dispBlanksAs val="gap"/>', new)
                    assert k == 1, 'plotVisOnly not found in ' + item.filename
                changed.append((item.filename, n))
                data = new.encode('utf-8')
        zout.writestr(item, data)
shutil.move(tmp, dst)
print('removed empty points:', changed)
