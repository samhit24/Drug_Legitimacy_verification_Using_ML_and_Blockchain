import pandas as pd
from web3 import Web3
import json
import time

# ✅ Connect to Ganache
ganache_url = "http://127.0.0.1:8545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

if not web3.is_connected():
    print("❌ Unable to connect to Ganache")
    exit(1)
else:
    print("✅ Connected to Ganache")

# ✅ Load ABI
with open("contract_abi.txt", "r") as f:
    abi = json.load(f)

# ✅ Deployed contract address
contract_address = "0x8CdaF0CD259887258Bc13a92C0a6dA92698644C0"
contract = web3.eth.contract(address=contract_address, abi=abi)

# ✅ Use first account with sufficient ETH
account = "0x627306090abaB3A6e1400e9345bC60c78a8BEf57"

# ✅ Load your dataset
df = pd.read_csv("medicine_with_batch_ids_10k.csv")

# ✅ Register all batches
for idx, row in df.iterrows():
    try:
        batch_id = str(row["batch_id"])
        product_name = str(row["Product Name"])
        status = True  # Legitimate

        tx_hash = contract.functions.addBatch(batch_id, product_name, status).transact({
            "from": account
        })

        web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ Registered {batch_id} — {product_name}")

        time.sleep(0.05)  # Gentle delay

    except Exception as e:
        print(f"❌ Error on {batch_id}: {e}")

print("🎉 All batches registered successfully!")
