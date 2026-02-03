# FlowLang (مسير) – Prototype v0.1

FlowLang هي لغة نطاق خاص (DSL) لتنسيق الأوامر المهنية ضمن "مسير" يحتوي على نقاط تفتيش Checkpoints، مع هياكل نظام: Teams، Chains، وProcess Trees.

هذا النموذج الأولي يتضمن:
- Grammar (Lark) مع دعم القوائم والقواميس وحقول مثل `J3.confidence`.
- AST مبسّط.
- Semantic Analyzer لقواعد التوافق الدلالية.
- Runtime بسيط لتنفيذ `flow` و`checkpoint` و`flow.back_to` و`chain.touch`.
- مثال `examples/example1.flow`.

## المتطلبات

- Python 3.10+
- `lark` (مثبتة عبر `requirements.txt`)

## التثبيت

```bash
pip install -r requirements.txt
```

## التشغيل

```bash
python scripts/run.py examples/example1.flow
```

## المكونات

- `flowlang/grammar.lark`: قواعد النحو.
- `flowlang/ast.py`: تعريف عقد AST.
- `flowlang/types.py`: نظام أنواع خفيف.
- `flowlang/parser.py`: المحلل (Lark) وتحويل إلى AST.
- `flowlang/semantic.py`: تحقق دلالي (teams/kinds, checkpoints, chain nodes).
- `flowlang/runtime.py`: محرك المسير والتنفيذ، مع حالات `chains/processes`.
- `flowlang/errors.py`: استثناءات مخصّصة.

## حدود النموذج الأولي
- التنفيذ للأوامر `search/try/judge/ask` وهمي ويعيد كائنات بسيطة مع `confidence` عند الحاجة.
- لا يوجد FFI حقيقي حتى الآن.
- `process` و`audit` تُسجّل فقط، مع دعم عمليات نظامية (`mark/expand/collapse`).

## خصائص السلاسل والأشجار النظامية

- Chain (`chain ... { ... }`):
  - `nodes`: عقد السلسلة.
  - `propagation: causal(...)`: إعدادات الانتشار (decay/backprop/forward/cap).
  - `labels: { key: "value" }`: وسوم مفتاحية.
  - `constraints: { key: value; ... }`: قيود رقمية/منطقية/نصية.
  - استدعاءات نظامية ضمن `flow`:
    - `ModelUpdateChain.set_label(key, value)`
    - `ModelUpdateChain.get_label(key)` (يرجع في المتغير `_`)
    - `ModelUpdateChain.set_constraint(key, value)`
    - `ModelUpdateChain.propagate(node, effect)`

- Process Tree (`process ... { ... }`):
  - `branch "Name" -> ["ChildA", "ChildB"];`
  - `node "Name" { prop: value; ... };`
  - `policy: { key: value; ... };`
  - `audit: enabled;` ويسمح داخل الـ flow: `ProductTree.audit()`
  - استدعاءات نظامية ضمن `flow`:
    - `ProductTree.mark(node, status)`
    - `ProductTree.expand(parent, [children...])`
    - `ProductTree.collapse(node)`

## التوسعات القادمة
- نظام أنواع أكثر ثراءً (Records/Union)، وربط أدوات حقيقية.
- جدولة فرق متقدمة (weighted/priority).
- تغطية نحوية أوسع وملفات اختبارات.
