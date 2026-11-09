"""app/template.html 에 app/data.json 을 넣어 app/index.html(단일 파일)을 만든다."""
import json
t = open("app/template.html", encoding="utf-8").read()
d = json.load(open("app/data.json", encoding="utf-8"))
open("app/index.html", "w", encoding="utf-8").write(t.replace("__DATA__", json.dumps(d, ensure_ascii=False)))
print("app/index.html")
