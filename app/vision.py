"""SEES: the vision step (D43 pattern with Nemotron VL). Code lists the options, the VL model picks one, and we read
a probability for every option, like the video: "damaged in transit 0.92 / wrong item 0.03 / changed mind 0.01".

async def sees(key, image, options: dict[label, description], text=None, priority="interactive") -> dict
    -> {"label", "confidence", "scores": {label: p}, "model", "usage", "by"}
    image: local path, data: URL or http(s) URL. "unclear" is always added as an option.
def image_parts(messages) -> list[str]     # image URLs (data: or http) from OpenAI-style messages
def to_data_url(path) -> str

Scoring (verified live on Token Factory, 2026-10-27):
- Super-VL thinks by default (400 reasoning tokens, empty content) -> config.NO_THINK is required.
- structured_outputs.choice is enforced, and logprobs are post-mask: disallowed tokens come back at -9999 and forced
  tokens at 0. So the first generated token's top_logprobs is a distribution over the options' first tokens.
  One call gives every option's score; no per-option calls.
"""
import base64, json, math, mimetypes, re

from . import config, limits

MODEL = "nvidia/Nemotron-3.5-Super-VL"
UNCLEAR = "unclear"
ISSUES = {  # the photo test set's labels (demo/photo_emails.py); first tokens are all distinct
    "damaged_in_transit": "arrived damaged by shipping: crushed, torn or wet box or packaging, item crushed or broken in transit",
    "defective_product": "a flaw in the item itself: hole, tear, broken part, stain, bad stitching or seam, peeling, falling apart",
    "wrong_item": "a different item, color or size than ordered was sent",
    "not_as_described": "the right item, but it looks different from the listing photos or description (color, material, style)",
    "changed_mind_or_fit": "nothing wrong with the item, but it does not fit or the customer does not like it",
    "no_issue": "the item looks fine; the customer is happy or just sharing a photo",
}


def to_data_url(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def image_parts(messages):
    out = []
    for m in messages or []:
        c = m.get("content")
        for p in c if isinstance(c, list) else []:
            if isinstance(p, dict) and p.get("type") == "image_url":
                u = p.get("image_url")
                u = u.get("url") if isinstance(u, dict) else u
                if isinstance(u, str) and u:
                    out.append(u)
    return out


def scores_from_first_token(top, labels):
    """top: top_logprobs of the first generated token. Each token's probability goes to the options it starts.
    ponytail: a token shared by several options (e.g. "d" for damaged/defective) is split evenly; pick labels
    with distinct first words, or score each option with its own call if that ever matters."""
    s = dict.fromkeys(labels, 0.0)
    live = [((t.get("token") or "").strip().lower(), t.get("logprob")) for t in top or []]
    live = [(tok, lp) for tok, lp in live if tok and lp is not None and lp > -50]  # -9999 = masked by the constraint
    for tok, lp in live:
        hit = [l for l in labels if l.lower().startswith(tok)]
        # "no" starts both no_issue and not_as_described; if "not" is also on the list, "not" owns not_as_described
        hit = [l for l in hit if not any(len(u) > len(tok) and l.lower().startswith(u) for u, _ in live)] or hit
        for l in hit:
            s[l] += math.exp(lp) / len(hit)
    z = sum(s.values())
    return {l: round(v / z, 4) for l, v in s.items()} if z else s


async def sees(key, image, options, text=None, priority="interactive"):
    options = {**options, UNCLEAR: options.get(UNCLEAR, "the photo does not show enough to tell")}
    labels = list(options)
    url = image if re.match(r"(?i)(data:|https?://)", str(image)) else to_data_url(image)
    listing = "\n".join(f"- {k}: {v}" if v else f"- {k}" for k, v in options.items())
    content = [{"type": "text", "text": f"Customer message: {text}" if text else "The customer sent only this photo."},
               {"type": "image_url", "image_url": {"url": url}}]
    body = {"model": MODEL, "temperature": 0, "max_tokens": 20, "logprobs": True, "top_logprobs": 20,
            "structured_outputs": {"choice": labels}, **config.NO_THINK,
            "messages": [{"role": "system", "content":
                          "You look at photos customers send to an outdoor clothing and footwear store. Judge what the "
                          f"photo itself shows. Pick the one option that fits best:\n{listing}\n\nReply with the label only."},
                         {"role": "user", "content": content}]}
    status, data, _, model, _ = await limits.call(key, body, priority)
    ok = status == 200 and isinstance(data, dict) and data.get("choices")
    if not ok:
        return {"label": UNCLEAR, "confidence": 0.0, "scores": {}, "model": model, "usage": None, "by": "error",
                "error": f"{status}: {str(data)[:200]}"}
    ch = data["choices"][0]
    out = (ch["message"].get("content") or "").strip()
    label = out if out in options else UNCLEAR
    lp = (ch.get("logprobs") or {}).get("content") or []
    scores = scores_from_first_token(lp[0].get("top_logprobs") if lp else None, labels)
    if not any(scores.values()):  # no logprobs came back: fall back to the chosen path's own probability
        scores = {l: 0.0 for l in labels}
        scores[label] = round(math.exp(sum(t.get("logprob") or 0 for t in lp)), 4) if lp else 0.0
    return {"label": label, "confidence": scores.get(label, 0.0), "scores": scores, "model": model,
            "usage": data.get("usage"), "by": "vl"}


# Photo-vs-request check (live: "my boots are falling apart" + a photo of a PlayStation item -> must push back).
# Only for messages about an order/product problem; FAQ photos skip it.
NEEDS_MATCH = re.compile(r"(?i)\b(order|return|refund|replace|replacement|exchange|defect\w*|broken|broke|damag\w*|torn|"
                         r"rip\w*|hole|stain\w*|crack\w*|falling apart|fell apart|wrong|missing|boots?|shoes?|sneakers?|"
                         r"jackets?|coats?|shirts?|pants|jeans|shorts|socks|gloves?|hats?|tents?|backpacks?|bags?|"
                         r"sleeping bag|vest|hoodie|fleece|sweater|item|package|box|parcel)\b")
VERDICT = {"yes": "match", "no": "mismatch", "unclear": "unclear"}  # distinct first tokens ("match"/"mismatch" share "m")
MATCH_FMT = {"type": "json_schema", "json_schema": {"name": "photo_match", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["item", "shows", "same"],
    "properties": {"item": {"type": "string", "maxLength": 40}, "shows": {"type": "string", "maxLength": 60},
                   "same": {"type": "string", "enum": list(VERDICT)}}}}}


def needs_match(text):
    return bool(text and NEEDS_MATCH.search(text))


def value_top(lp, content, key):
    """top_logprobs at the first token of `key`'s string value in a JSON reply, with the part of that token before the
    value (e.g. the opening quote) stripped from every alternative, so scores_from_first_token can read it."""
    m = re.search(rf'"{key}"\s*:\s*"', content)
    if not m or not lp:
        return None
    pos, start = m.end(), 0
    for t in lp:
        tok = t.get("token") or ""
        if start + len(tok) > pos:
            pre = tok[:max(0, pos - start)]
            return [{**a, "token": a["token"][len(pre):]} for a in t.get("top_logprobs") or [t]
                    if (a.get("token") or "").startswith(pre)]
        start += len(tok)
    return None


async def matches(key, image, text, priority="interactive"):
    """Does the photo show what the customer describes? One VL call: {"item", "shows", "same": yes|no|unclear}.
    -> {"verdict": match|mismatch|unclear, "confidence", "scores", "item", "shows", "model", "usage"}"""
    url = image if re.match(r"(?i)(data:|https?://)", str(image)) else to_data_url(image)
    body = {"model": MODEL, "temperature": 0, "max_tokens": 80, "logprobs": True, "top_logprobs": 20,
            "response_format": MATCH_FMT, **config.NO_THINK,
            "messages": [{"role": "system", "content":
                          "You check photos customers send to an outdoor clothing and footwear store. item: the product "
                          "the customer's message is about, in 1-4 words (e.g. 'boots'). shows: what the photo actually "
                          "shows, in a few words. same: 'yes' if the photo shows that product or its situation (the item, "
                          "its damage, or its box or package), 'no' if it clearly shows something else, 'unclear' if you "
                          "cannot tell."},
                         {"role": "user", "content": [{"type": "text", "text": f"Customer message: {text}"},
                                                      {"type": "image_url", "image_url": {"url": url}}]}]}
    status, data, _, model, _ = await limits.call(key, body, priority)
    out = {"verdict": UNCLEAR, "confidence": 0.0, "scores": {}, "item": "", "shows": "", "model": model,
           "usage": data.get("usage") if isinstance(data, dict) else None}
    if not (status == 200 and isinstance(data, dict) and data.get("choices")):
        return {**out, "error": f"{status}: {str(data)[:200]}"}
    ch = data["choices"][0]
    content = ch["message"].get("content") or ""
    try:
        j = json.loads(content)
    except ValueError:
        return {**out, "error": f"bad json: {content[:200]}"}
    same = j.get("same") if j.get("same") in VERDICT else "unclear"
    lp = (ch.get("logprobs") or {}).get("content") or []
    s = scores_from_first_token(value_top(lp, content, "same"), list(VERDICT))
    if not any(s.values()):  # no logprobs: the chosen answer only, unscored
        s = dict.fromkeys(VERDICT, 0.0)
    scores = {VERDICT[k]: v for k, v in s.items()}
    return {**out, "verdict": VERDICT[same], "confidence": scores[VERDICT[same]], "scores": scores,
            "item": str(j.get("item") or "")[:40], "shows": str(j.get("shows") or "")[:60]}
