# Safe Hypercore Writer trials

## Trial 3 — HyperEVM mainnet (2026-03-09)

**Network:** HyperEVM mainnet (chain 999)
**Result:** Steps 1–6 succeeded, step 7 (revalue) failed due to stale vault equity cache (now fixed with `bypass_cache` parameter)

### Addresses

| Role | Address |
|------|---------|
| Deployer (EOA) | `0x4E6B7f7aFB2E23Bf9355c10e4454f73E6E6F3D9c` |
| Safe | `0x7bEfA4a93A5c19b578b14A1BC5c4CaBb0B8D7991` |
| Lagoon Vault | `0x766089071255274ad4E5f91d2b486e0A1eCaC20C` |
| TradingStrategyModuleV0 | `0xE0B3a42c3e34Da277A5a840Bf86B6bd48E9D5c39` |
| UniswapLib | `0xe2cE850D99506a7aCB631C5d441C39b09D573014` |
| HypercoreVaultLib | `0x9Fc585Ec646C9a4ACB729a7c0e4EB9f9C1640b87` |
| Hypercore vault (HLP) | `0xdfc24b077bc1425AD1DEA75bCB6f8158E10Df303` |
| CoreDepositWallet | `0x6B9E773128f453f5c2C60935Ee2DE2CBc5390A24` |
| CoreWriter precompile | `0x3333333333333333333333333333333333333333` |

### Deployer nonces

| Step | Nonce range |
|------|-------------|
| Deployment start | 479 |
| Deployment end (guard + whitelist) | 490 |
| Vault init + fund + settle | 493–502 |
| Activation (approve + depositFor) | 503–504 |
| Phase 1 multicall | 505 |
| Phase 2 multicall | 506 |

### Safe salt nonce

`1243637365`

### Transaction hashes

| Step | Tx hash | Block | Notes |
|------|---------|-------|-------|
| Activation: USDC approve | `b6536c2bd9d7930f617ace56439eec90d61da15d64bd22c1661f1d7805260e74` | — | Approve USDC to CDW |
| Activation: depositFor | `6fe75572ee9d5943df6c47cdecb67e3a6db0a54782602684d3e7cd367cb2fa45` | — | 2 USDC depositFor → activates Safe on HyperCore |
| Phase 1 multicall | `35536ee4e432ca92947d4252e719f92975f2c88095950fc69fa7db7a1e06d582` | 29,311,509 | approve + CDW.deposit (5 USDC) |
| Phase 2 multicall | `f07ae3d911c84c3d3f8fd8fce2913057699627b70777fc6551891d57bc0808a8` | — | transferUsdClass + vaultTransfer |
| Vault fund (approve) | `1b393dfcea5f0015c68d6f70c6750054555017717a79be181a25a9421901d969` | — | 7 USDC approve for vault deposit |
| Vault fund (deposit request) | `e8c67d4a361a457813c1cb3948e219090b39e9a9ee08c24af3d83b76a13b5e2f` | — | requestDeposit to Lagoon |
| Vault settle (valuation) | `407a5125b7a3866f275c946215b418cfc8e491e978cc9f645f6eaa4ad1fad8bb` | — | Post valuation |
| Vault settle (settleDeposit) | `be5855845fcbabea766ba190cfb50079f96e35471935889d4bbbb778ae4435a1` | — | settleDeposit |
| Manual phase 2 (post-failure) | `b464e674a14a2ed291111149de41414f59a927d6369f504ba7340e5bd169d905` | — | Manual retry of phase 2 (147,539 gas) |

### Timeline

```
20:35:35  Deployment starts (nonce 479)
20:39:20  Deployment complete (Safe + Lagoon vault + module + guard)
20:40:16  Vault init: post initial valuation
20:40:33  Vault settle deposits
20:41:39  Vault settled (7 USDC, block 29,311,431)
20:42:08  Strategy cycle starts
20:42:19  Safe NOT activated on HyperCore → activation flow begins
20:42:32  Activation: approve tx broadcast
20:42:34  Activation: depositFor tx broadcast
20:42:46  Safe activated on HyperCore (confirmed via coreUserExists precompile)
20:42:47  Phase 1 multicall broadcast (5 USDC, nonce 505)
20:42:52  Phase 1 confirmed (block 29,311,509)
20:42:54  EVM escrow cleared (1 poll)
20:42:56  Phase 2 multicall confirmed → equity = 5 USDC
20:43:37  Revalue failed: equity=0 (stale cache), valuation_price=0.0
```

### Amounts

| Item | Amount |
|------|--------|
| USDC funded to Safe | 7 USDC |
| Activation cost | 2 USDC |
| Net deposit to HyperCore | 5 USDC |
| Vault equity after deposit | 5 USDC |
| Vault equity at revalue (cached) | 0 USDC (stale — cache bug) |

### Explorer links

- Safe: https://www.hyperscan.com/address/0x7bEfA4a93A5c19b578b14A1BC5c4CaBb0B8D7991
- Lagoon vault: https://www.hyperscan.com/address/0x766089071255274ad4E5f91d2b486e0A1eCaC20C
- Module: https://www.hyperscan.com/address/0xE0B3a42c3e34Da277A5a840Bf86B6bd48E9D5c39
- Phase 1 tx: https://www.hyperscan.com/tx/0x35536ee4e432ca92947d4252e719f92975f2c88095950fc69fa7db7a1e06d582
- Phase 2 tx: https://www.hyperscan.com/tx/0xf07ae3d911c84c3d3f8fd8fce2913057699627b70777fc6551891d57bc0808a8

### Notes

- Activation via `depositFor` works on **mainnet** but NOT on testnet (see [web3-ethereum-defi#813](https://github.com/tradingstrategy-ai/web3-ethereum-defi/issues/813))
- The revalue failure was caused by a 15-minute module-level cache in `fetch_user_vault_equity()` returning stale equity=0. Fixed by adding `bypass_cache=True` parameter.
- HLP vault minimum deposit is 5 USDC
