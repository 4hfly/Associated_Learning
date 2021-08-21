# news of the project

## coding issues

    1. argparse
        - 有打算繼續使用這方法嗎？用 json 控制的話，你之前是怎麼維護那個 json 的？
        - 我之前有放個 `load_parameters()`, `save_parameters()` 的 code 在 `model.py` 裡，這個應該可以取代 argparse 吧？就用全域變數的方法。
    2. 我習慣 `def:`, `class:`, `for:`, `with:` 下面會空白一行，除非它裡面只包了一行（就像 C 的寫法）

## requirements (libs version)

    1. python: 3.8.10 (newest ver. is 3.8.11)
        - 3.9 和 3.10 我沒研究過。
    2. pytorch: 1.8.1
        - anaconda 上最新好像就這個版本，官網 stable 是到 1.9.0。
    3. CUDA: 11.3(newest 11.4)
