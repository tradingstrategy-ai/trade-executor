# Cross-chain vault test matrix — 2026-07-25 rerun

Full 129-vault rerun after integrating eth-defi #1374, #1375 and #1376 and the
executor-side typed-result work.

- trade-executor: `754d0676`
- eth-defi: master `0755b7398` (#1376 cSuperior redemption preflight)
- mode: `--auto-simulated --settle-async-on-anvil`, amount 1.0
- machine-readable: `cross-chain-vault-test-2026-07-25-rerun.report.json`
- vault id list: `cross-chain-vault-test-vault-ids.txt`

Run note: executed as 20 batched rounds against one shared state file after
three monolithic attempts died on transient in-fork RPC disconnects. Same
commits throughout, so rows are directly comparable, but it was not a single
process.

## Summary vs baseline

| Category | Rerun | Baseline | Δ |
|---|---:|---:|---:|
| Success (simulated) | 55 | 43 | +12 |
| Deposit closed | 45 | 51 | −6 |
| Whitelisting needed | 14 | 14 | 0 |
| **Gaps** | **15** | 21 | **−6** |
| **Forbidden results** | **2** | 12 | **−10** |

No regressions: nothing moved from success/deposit-closed/whitelisting into a
gap. All six improvements are Lagoon Finance (three `broadcast failed` and three
`simulation unsupported async` now complete a full lifecycle).

Part of the `deposit_closed` → `success` movement is live on-chain admission
drift rather than a code effect; the Lagoon +6 is the code-attributable part.

## Remaining 15 gaps

13 of 15 are typed current-state results carrying machine-readable reasons.
Only two are genuine failures, both cross-chain satellite redemptions where the
vault holds no immediate underlying liquidity:

| Protocol | # | Nature |
|---|---:|---|
| Ember | 5 | 4× typed false-capability, 1× `below_minimum` |
| cSigma Finance | 2 | both `redemption_capacity_limited` |
| Gains Network | 2 | both `redemption_window_closed` |
| Plutus | 1 | `simulation_unsupported_async`, role-gated reason |
| Accountable | 1 | `below_minimum` on the exact Monad address |
| Upshift | 1 | `incompatible_deposit_asset` |
| YieldNest | 1 | `redemption_unavailable` |
| **IPOR Fusion** | **1** | **`transaction_reverted`** — Autopilot USDC Morpho (Base) |
| **40acres** | **1** | **`transaction_reverted`** — Pharaoh USDC (Avalanche) |

## Full matrix

| # | vault id | vault | chain | protocol | status | detail |
|--:|---|---|---|---|---|---|
| 1 | `1-0xd5d097f278a735d0a3c609deee71234cac14b47e` | cSigma USD | Ethereum | cSigma Finance | redemption capacity limited | Existing terminal result; use --rerun to retest |
| 2 | `1-0x438982ea288763370946625fd76c2508ee1fb229` | cSuperior Quality Private Credit USDC | Ethereum | cSigma Finance | redemption capacity limited | Existing terminal result; use --rerun to retest |
| 3 | `1-0x2b13311fd553e74b421d4ccc96e348f71e179dcf` | Ember Apollo ACRED | Ethereum | Ember | below minimum | Existing terminal result; use --rerun to retest |
| 4 | `1-0x9be9294722f8aad37b11a9792be2c782182cafa2` | Ember Earn | Ethereum | Ember | simulation unsupported async | Existing terminal result; use --rerun to retest |
| 5 | `1-0x0b9342c15143e8f54a83f887c280a922f4c48771` | Ember Polymarket | Ethereum | Ember | simulation unsupported async | Existing terminal result; use --rerun to retest |
| 6 | `1-0xf3190a3ecc109f88e7947b849b281918c798a0c4` | Ember Third Eye | Ethereum | Ember | simulation unsupported async | Existing terminal result; use --rerun to retest |
| 7 | `1-0x373152feef81cc59502da2c8de877b3d5ae2e342` | Ember UDL | Ethereum | Ember | simulation unsupported async | Existing terminal result; use --rerun to retest |
| 8 | `1-0x95b2ed8f821570f85fd0e3e6e7088c6296587088` | BL USDC WSR Loop  | Ethereum | IPOR Fusion | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 9 | `1-0xf8f226da66244f89e70c5b5d1a5c5b0d505eb1d8` | Bitcoin Dollar USDC | Ethereum | IPOR Fusion | success (simulated) | Existing terminal result; use --rerun to retest |
| 10 | `1-0x0b45a1e71a8a09f5d382fed27202d50ed983aaf3` | Hyperithm mHYPER Looping | Ethereum | IPOR Fusion | success (simulated) | Existing terminal result; use --rerun to retest |
| 11 | `1-0x43ee0243ea8cf02f7087d8b16c8d2007cc9c7ca2` | IPOR USDC Ethereum Optimizer | Ethereum | IPOR Fusion | success (simulated) | Existing terminal result; use --rerun to retest |
| 12 | `1-0xb0f56bb0bf13ee05fef8cd2d8df5ffdfcac7a74f` | TAU InfiniFi Pointsmax | Ethereum | IPOR Fusion | success (simulated) | Existing terminal result; use --rerun to retest |
| 13 | `1-0x63103375659d0aa94e9f35df15be01a3dd1ae9c0` | TAU Lending Optimizer | Ethereum | IPOR Fusion | success (simulated) | Existing terminal result; use --rerun to retest |
| 14 | `1-0x43a32d4f6c582f281c52393f8f9e5ace1d4a1e68` | TAU Yield Bond ETF | Ethereum | IPOR Fusion | success (simulated) | Existing terminal result; use --rerun to retest |
| 15 | `1-0x888e1d3c509c80e24cab8a4872e164b7e5a6eb10` | TESS USDC Ethena Loop Vault | Ethereum | IPOR Fusion | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 16 | `1-0xc825779c89120eeef746c51130b362478e181d39` | TESS USDC Lending Optimiser | Ethereum | IPOR Fusion | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 17 | `1-0x4c5a611694c426cae9335d53e95b885090cf8c31` | TESS USDC wsrUSD Loop | Ethereum | IPOR Fusion | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 18 | `1-0x32f07401eb177f2c0fc4f95f3928050d88dae7ed` | TESS sUSDe PYUSD (USDC) Loop Vault | Ethereum | IPOR Fusion | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 19 | `1-0xc2a119ea6de75e4b1451330321cb2474eb8d82d4` | Tesseract USDC Lending Optimizer | Ethereum | IPOR Fusion | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 20 | `1-0xbb30c3b6046debcbe941281218d18dec8ecebeb5` | 1212.Stable | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 21 | `1-0x3be67ba2d3fec744d1d2b5d564c83f57372578e4` | AltaETF | Ethereum | Lagoon Finance | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 22 | `1-0x8417430a31851ae0a36a854394227c5d86be8fc9` | Ammalgam USDC Vault | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 23 | `1-0xdcd0f5ab30856f28385f641580bbd85f88349124` | Autonomous Liquidity USD | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 24 | `1-0x9fdbaaa76194d56e49cade12c1f216f47d2b865e` | Block4Block | Ethereum | Lagoon Finance | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 25 | `1-0xfab0f56c28e3f874b15922b213e696f37b670916` | Coinshift Conservative USPC | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 26 | `1-0x7d2c2f54792ad72cb834d298f542145b06b703cb` | DeTrade Morpho X-Chain USDC | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 27 | `1-0xf10801bcc3deaf467fb8b3dbb7430111822e6dab` | Der USDC | Ethereum | Lagoon Finance | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 28 | `1-0xba6cfe8a9d199cd7f3e50114c4e4ec66f2d52c87` | Der base USDC | Ethereum | Lagoon Finance | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 29 | `1-0x06973fbca7c589d10dfbe45d694dce634bff6165` | FLEXIBLE CRYPTO FUND | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 30 | `1-0x59b7942f7d2afd085691ce65c152e0d38d4eff22` | Gami Capital lvlUSD | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 31 | `1-0xdae854d0896ad2fee335689a3f7b4a95fd1a3e46` | Gami USDC | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 32 | `1-0xca790385506b790554571cbc9da73f0130cdcfd5` | Hub Capital USDC vault | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 33 | `1-0xa00f63e85b3d242568a9edecb48f5e2cf879b07b` | Moon Digital AM USDC | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 34 | `1-0x14b8de97d078cabf38ba3d0b7e067618f0e8ab7d` | Mt Pelerin x Montaigne - Strategy pool | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 35 | `1-0x28663161f9fa2963eb6102b88a741e195e974df6` | Mt Pelerin – USD strategy pool | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 36 | `1-0xef39d77c7fb6224ac974c5fa4e3151a6c6ce9594` | Muchacho USDC | Ethereum | Lagoon Finance | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 37 | `1-0xb993c32f578e5156369330787cf8c8fe033bf40e` | Noon STS USDC | Ethereum | Lagoon Finance | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 38 | `1-0xcb58582b0d52ce5feecb06ba9ce66598b0d57886` | Strada USDC | Ethereum | Lagoon Finance | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 39 | `1-0xa96bc6e084aad6976d25df9431525ed2c4d3cae4` | Syntropia <> Resupply | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 40 | `1-0x8df3deba711ae4a9af16cbca5e4fbb1402f036d5` | Syntropia Boosted | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 41 | `1-0xd17049ed25d8f99fe3bfd10cef2263da9995cfd8` | Syntropia USDC | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 42 | `1-0x1b2cb79a4564206f53ba80b4d780f251b4ae6765` | Syntropia USDC Core | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 43 | `1-0xce0b790ae0d8cf91e01f3fb69025e14569b574f3` | Tulipa USDC | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 44 | `1-0xb4a4c9a736f91e2694c6b921445eef3e3585a591` | Zharta RWA Prime USDC | Ethereum | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 45 | `1-0x175ea882b492c9b7a6d5852fe9da560dc7af1c72` | pyUSDC | Ethereum | Lagoon Finance | whitelisting-needed | Existing terminal result; use --rerun to retest |
| 46 | `1-0xbeef01735c132ada46aa9aa4c54623caa92a64cb` | Steakhouse USDC | Ethereum | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 47 | `1-0x33ffc177a7278ff84aab314a036bc7b799b7cc15` | Arche USD | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 48 | `1-0x7b5a0182e400b241b317e781a4e9dedfc1429822` | Katana Pre-Deposit USDC | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 49 | `1-0xf470eb50b4a60c9b069f7fd6032532b8f5cc014d` | Katana Pre-Deposit USDC | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 50 | `1-0xf20b02131c45b22e147c98e30cc889a20dc8a00d` | Katana Pre-DepositUSDC | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 51 | `1-0xaca399117ac588e1f48398d34eca76cdb1e45fa5` | Ondo aggregator | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 52 | `1-0x4dd0fe8549641a04d7ab4f37dbb541ae7dbb2838` | Silo-LRT USDC yVault | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 53 | `1-0xf6e2d36c489e5b361cdc962d4568cea663ad5ddc` | StrategyGearboxLenderUSDC | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 54 | `1-0x46e9893422b9ae9246793489433f72c548cb2455` | Sturdy USDC-taoUSD bridge vault | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 55 | `1-0xb73b7dc2d967473ad669e35186efee3335e30eb9` | Teller USDC | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 56 | `1-0xc62fc9b0bb3d9c7a47a6af1ed30d7a4c74e37774` | Test yvUSD | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 57 | `1-0x696d02db93291651ed510704c9b286841d506987` | USD yVault | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 58 | `1-0x4c0e4d3cb62b91afbbf1fe8e830f98a513c7234b` | USD3 Pendle PT Maxi | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 59 | `1-0x62ebe2ca290db3b649c390847f8204196771b438` | USD3 Pendle PT Maxi | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 60 | `1-0x697c54a84d83f37380d034e2bfc6f7ce8d89f4ee` | USDC Meta Vault | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 61 | `1-0xbe53a109b494e5c9f97b9cd39fe969be68bf6204` | USDC-1 yVault | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 62 | `1-0xae7d8db82480e6d8e3873ecbf22cf17b3d8a7308` | USDC-2 yVault | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 63 | `1-0x8670120c32de7bc990e0fe3bbd04704e98492f0a` | Usual aggregator | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 64 | `1-0x7bb36e40a7b08f653ddc24e2e1181559f4d52f2a` | Yeet It USD | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 65 | `1-0x2df6c1602528de8b8a5c72baf6e70295b3a64142` | sUSDS Pendle PT Maxi | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 66 | `1-0x9cfb40acedac259b1d23e790f6c6d0c3898361ad` | usdc-vault | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 67 | `1-0x01ba69727e2860b37bc1a2bd56999c1afb4c15d8` | YieldNest RWA MAX | Ethereum | YieldNest | redemption unavailable | Existing terminal result; use --rerun to retest |
| 68 | `8453-0xb99b6df96d4d5448cc0a5b3e0ef7896df9507cf5` | Aerodrome USDC | Base | 40acres | success (simulated) | Existing terminal result; use --rerun to retest |
| 69 | `8453-0xad20523a7dc37babc1cc74897e4977232b3d02e5` | gTrade (Gains Network USDC) | Base | Gains Network | redemption window closed | Existing terminal result; use --rerun to retest |
| 70 | `8453-0x0d877dc7c8fa3ad980dfdb18b48ec9f8768359c4` | Autopilot USDC Base | Base | IPOR Fusion | success (simulated) | Existing terminal result; use --rerun to retest |
| 71 | `8453-0xd6701905c59ee618dc36dc747506bce0a4ac760a` | Autopilot USDC Morpho (Base) | Base | IPOR Fusion | transaction reverted | Existing terminal result; use --rerun to retest |
| 72 | `8453-0x45aa96f0b3188d47a1dafdbefce1db6b37f58216` | IPOR USDC Lending Optimizer Base | Base | IPOR Fusion | success (simulated) | Existing terminal result; use --rerun to retest |
| 73 | `8453-0x1166250d1d6b5a1dbb73526257f6bb2bbe235295` | yoUSD Looopeer | Base | IPOR Fusion | success (simulated) | Existing terminal result; use --rerun to retest |
| 74 | `8453-0xb09f761cb13baca8ec087ac476647361b6314f98` | 722Capital-USDC | Base | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 75 | `8453-0x8092ca384d44260ea4feaf7457b629b8dc6f88f0` | DeTrade Core USDC | Base | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 76 | `8453-0x2bff679b1a9fbcc202316c1402172747ba2fbf56` | For Yield v2 | Base | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 77 | `8453-0xd5c22fa3f7ee979ed7c28e36669b29797ab277e4` | GUSDC | Base | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 78 | `8453-0xf56bfe07b8d6e6d74258cdb6969a633629b06b08` | MSA - Portefeuille Dynamique  | Base | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 79 | `8453-0x4efc07dca8697792119484af33549f33ab11bf3c` | MoneyFi FlowForge Vault | Base | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 80 | `8453-0x63b04d3ce2c14f6d308657ab73ac92fc1a0b1075` | RB Capital Yield | Base | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 81 | `8453-0xbe7db44f4ce20dac83b578b94fd35087f66e9754` | TruMarket | Base | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 82 | `8453-0x94d886d25729150adfa20210f9b94cefe0b3d132` | Azur USDC | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 83 | `8453-0xcb2f26898c0893c0bdd5cf76417cf9b2258af0ed` | Chimi USDC Vault | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 84 | `8453-0x0c352ab624313ae28ed4073dc5d469eb36e164c6` | Chimi USDC Vault Turbo | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 85 | `8453-0x50fd1e6e0e2153c2b26ebbcd9bcded4639a1aae3` | MUV002 - Flagship USDC | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 86 | `8453-0xfdb431e661372fa1146efb70bf120ecded944a78` | Moonwell USDC Lender WETH Borrower | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 87 | `8453-0x945df73d55557ea23c0c35cd350d8de3b745287e` | Moonwell USDC Lender cbBTC Borrower | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 88 | `8453-0x7bb8b0b176199b3313642dde3421e38e548d6570` | RizVaultUSDC | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 89 | `8453-0x19f233b2953275196e6343f17b76da098c478e21` | Teller USDC | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 90 | `8453-0xb13cf163d916917d9cd6e836905ca5f12a1def4b` | True Yield Dollar | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 91 | `8453-0xd1468af648565f11393e4033cb0cd270b62495c9` | USDC BaseInvaders | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 92 | `8453-0xc3bd0a2193c8f027b82dde3611d18589ef3f62a9` | USDC Horizon yVault | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 93 | `8453-0x92a6f4cc1e26baa1beec168e3c346aedcb437f31` | Yeet It USD | Base | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 94 | `42161-0x75288264fdfea8ce68e6d852696ab1ce2f3e5004` | HYPE++ | Arbitrum One | D2 Finance | deposit closed | Existing terminal result; use --rerun to retest |
| 95 | `42161-0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0` | gTrade (Gains Network USDC) | Arbitrum One | Gains Network | redemption window closed | Existing terminal result; use --rerun to retest |
| 96 | `42161-0xd855296f9868ff659f3f359c90b7d005ec049228` | IPOR USDC Prime | Arbitrum One | IPOR Fusion | success (simulated) | Existing terminal result; use --rerun to retest |
| 97 | `42161-0x1723cb57af58efb35a013870c90fcc3d60174a4e` | Angmar Capital | Arbitrum One | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 98 | `42161-0x018282d5b510f00dcacb8f4a81c3901d2fc9da51` | Sandbox | Arbitrum One | Lagoon Finance | success (simulated) | Existing terminal result; use --rerun to retest |
| 99 | `42161-0xf84eaa0685626f84fe17bc6c3c9eb2ac8a90d3c1` | Plutus Dolomite vault | Arbitrum One | Plutus | deposit closed | Existing terminal result; use --rerun to retest |
| 100 | `42161-0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a` | Plutus Hedge Token | Arbitrum One | Plutus | simulation unsupported async | Existing terminal result; use --rerun to retest |
| 101 | `42161-0xe6dbfb035b44e94d07f7b3e4f6bfbf1c6e68e3d0` | MUV001 - Flagship USDC | Arbitrum One | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 102 | `42161-0x2e7aa06a0f0816de4b1a32a12b0ac4eb584bff2a` | RizVaultUSDC | Arbitrum One | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 103 | `42161-0xbe7d1b3cf19eb05ac557be14af24e093fadfd7c6` | Teller USDC | Arbitrum One | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 104 | `42161-0x6faf8b7ffee3306efcfc2ba9fec912b4d49834c1` | USDC-A yVault | Arbitrum One | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 105 | `42161-0x2b0b6376083c6e1f376c7439f328436a673f333c` | yPT-aUSDC (auto-rolling Pendle PT) | Arbitrum One | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 106 | `42161-0xdf9640332bcfd3b16cb80c1287ed04b875d9a384` | yPT-aUSDC (auto-rolling Pendle PT) | Arbitrum One | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 107 | `43114-0xc0485c4bafb594ae1457820fb6e5b67e8a04bcfd` | Blackhole USDC | Avalanche C-chain | 40acres | deposit closed | Existing terminal result; use --rerun to retest |
| 108 | `43114-0x124d00b1ce4453ffc5a5f65ce83af13a7709bac7` | Pharaoh USDC | Avalanche C-chain | 40acres | transaction reverted | Existing terminal result; use --rerun to retest |
| 109 | `43114-0x9fd32fd5e32c6b95483d36c5e724c5c5250ce010` | ygamiUSDC | Avalanche C-chain | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 110 | `43114-0x7aca67a6856bf532a7b2dea9b20253f08bc9a85a` | ymevUSDC | Avalanche C-chain | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 111 | `1-0x0229db3921de71cfa43cfe9fb6a87b403647a9ae` | Hyperithm USDC Midcurve | Ethereum | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 112 | `1-0x069662d2588fcac24b5c209456db965d151556f0` | Apyx USDC | Ethereum | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 113 | `1-0x093272c07700d3ca5301c3bf9b3a392624179e2f` | Hyperithm USDC Degen | Ethereum | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 114 | `1-0x3cd3718f8f047aa32f775e2cb4245a164e1c99fb` | Hyperithm Euler USDC | Ethereum | Euler | success (simulated) | Existing terminal result; use --rerun to retest |
| 115 | `1-0x777791c4d6dc2ce140d00d2828a7c93503c67777` | Hyperithm USDC Degen | Ethereum | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 116 | `1-0x8aff4fe319c30475d27ec623d7d44bd5ecfe9616` | Hyperithm mHYPER | Ethereum | Euler | success (simulated) | Existing terminal result; use --rerun to retest |
| 117 | `1-0xabe418cc8c06d265e4eb009c02ea4b265eca7240` | Saturn USDC | Ethereum | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 118 | `1-0xba8704c18b55f60f5d84b53c3f39a0189a0965b3` | Morpho Hyperithm USDC Strategy | Ethereum | Yearn | success (simulated) | Existing terminal result; use --rerun to retest |
| 119 | `1-0xcdaea3dde6ce5969aa1414a82a3a681ced51ce72` | Hyperithm USDC Midcurve | Ethereum | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 120 | `143-0x78999cc96d2ba0341588c60ccb0e91c6c33cf371` | Hyperithm USDC Degen | Monad | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 121 | `143-0x7cd231120a60f500887444a9baf5e1bd753a5e59` | Hyperithm Delta Neutral Vault | Monad | Accountable | below minimum | Existing terminal result; use --rerun to retest |
| 122 | `143-0xa8665084d8cd6276c00ca97cbc0bf4bc9ae94c79` | Hyperithm USDC Degen | Monad | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 123 | `42161-0x4b6f1c9e5d470b97181786b26da0d0945a7cf027` | Hyperithm USDC | Arbitrum One | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 124 | `1-0x56bfa6f53669b836d1e0dfa5e99706b12c373ecf` | sky.money USDC Risk Capital | Ethereum | Morpho | success (simulated) | Existing terminal result; use --rerun to retest |
| 125 | `1-0x7130570bcefcedbe9d15b5b11a33006156460f8f` | USDC To SKY USDS Depositor | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 126 | `1-0x3d2467cbf82332dbfb38997cbc4d2192694d9490` | Morpho V2 Sentora PYUSD Convertor | Ethereum | Yearn | deposit closed | Existing terminal result; use --rerun to retest |
| 127 | `1-0x9bd52f2805c6af014132874124686e7b248c2cbb` | Sentora RLUSD | Ethereum | Euler | success (simulated) | Existing terminal result; use --rerun to retest |
| 128 | `1-0xab2726daf820aa9270d14db9b18c8d187cbf2f30` | Sentora PYUSD | Ethereum | Euler | success (simulated) | Existing terminal result; use --rerun to retest |
| 129 | `1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b` | Sentora USD Earn | Ethereum | Upshift | incompatible deposit asset | Existing terminal result; use --rerun to retest |