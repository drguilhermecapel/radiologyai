# Datasheets de datasets

Um datasheet por dataset que o projeto usa para treinar sondas, medir evidência ou verificar
viabilidade (ROADMAP §9.2). Nenhum pixel é versionado no git; os manifests com sha256 por arquivo
ficam no cache de dados e, para os conjuntos de avaliação, em `datasets/manifests/`.

| Dataset | Modalidade | Licença / termos (como obtidos) | Uso |
|---|---|---|---|
| [NIH ChestX-ray14](nih-chest-xray14.md) | RX tórax | termos do NIH | avaliação externa de pesos sem NIH no treino |
| [NIH ChestX-ray14 — rótulos adjudicados do Google](nih-cxr14-google.md) | RX tórax | imagens: termos do NIH; rótulos: licença **não conferida** | referência adjudicada (4 achados) sobre o teste oficial do NIH |
| [Open-i (Indiana)](openi.md) | RX tórax | imagens e laudos CC BY-NC-ND 4.0 | avaliação externa (itens pendentes) |
| [Shenzhen (NLM)](tb-shenzhen.md) | RX tórax | sem licença formal; não compartilhar fora do grupo | treino da sonda de TB |
| [Montgomery (NLM)](tb-montgomery.md) | RX tórax | idem | teste externo da sonda de TB |
| [FracAtlas](fracatlas.md) | RX musculoesquelético | CC BY 4.0 | sonda de fratura |
| [GRAZPEDWRI-DX](grazpedwri-dx.md) | RX punho pediátrico | CC BY 4.0 | sondas de gesso, metal, osteopenia |
| [BUSI](busi.md) | US mama | espelho não oficial; licença **não conferida** na fonte | sonda de lesão mamária (pendência) |
| [BUS-BRA](bus-bra.md) | US mama | CC BY 4.0 (Zenodo) + LICENSE no arquivo | candidato a treino/teste externo |
| [BrEaST](breast-lesions-usg.md) | US mama | CC BY 4.0 (TCIA) | candidato a teste externo |
| [Byra — esteatose](byra-fatty-liver.md) | US fígado | CC BY 4.0 (Zenodo) | sonda de esteatose |
| [Lesões hepáticas em US (Zenodo 7272660)](liver-lesion-us-7272660.md) | US fígado | CC BY 4.0 (Zenodo) | sonda de lesão hepática |
| [CAMUS](camus.md) | eco | CC BY-NC-SA 4.0, só pesquisa não comercial | referência de FE para o traçado assistido |
| [CQ500](cq500.md) | TC crânio | CC BY-NC-SA 4.0 (texto **não conferido** na fonte) | avaliação externa pendente do `totalseg-cerebral-bleed` |
| [CMMD](cmmd.md) | mamografia | CC BY 4.0 (TCIA, LICENSE no zip) | avaliação externa do mammoscreen (benigno x maligno) |
| [CBIS-DDSM](cbis-ddsm.md) | mamografia | CC BY 3.0 (TCIA, LICENSE no zip) | pré-treino do mammoscreen — **não** é externo para ele; só portões de RX |
