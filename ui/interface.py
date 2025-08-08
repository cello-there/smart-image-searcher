from pathlib import Path

def print_header(title: str):
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_result_list(results):
    for r in results:
        print(f"{r['score']:.4f}\t{r['path']}")