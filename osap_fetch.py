"""Download the full OSAP characteristic set via the bulk endpoint.

dl_all_signals pulls one ~1.6GB zip containing every signal, rather than 209
sequential requests.  Price/Size/STreversal are excluded: OSAP sources those
from CRSP directly and the package would require a WRDS login for them.
"""
import time, polars as pl
from openassetpricing import OpenAP

t0 = time.time()
oap = OpenAP()
names = sorted(oap.individual_signal_id_map["signal"].to_list())
names = [n for n in names if n not in ("Price", "Size", "STreversal")]
print(f"{len(names)} signals (CRSP-sourced trio excluded)", flush=True)

df = oap.dl_all_signals("polars", names)
df.write_parquet("cache/osap_all_raw.parquet")
print(f"[{time.time()-t0:.0f}s] {df.shape} -> cache/osap_all_raw.parquet", flush=True)

sign = oap.signal_sign.to_pandas().set_index("signal")["sign"].to_dict()
have = [c for c in df.columns if c not in ("permno", "yyyymm")]
print(f"  signs available for {sum(1 for c in have if sign.get(c) is not None)} of {len(have)}")
import json; json.dump({c: sign.get(c) for c in have}, open("cache/osap_signs.json", "w"))
print("  wrote cache/osap_signs.json")
