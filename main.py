from pathlib import Path
import yaml

CONFIG = Path("config.yaml")


def main():
    if not CONFIG.exists():
        print("config.yaml 파일이 없습니다.")
        return

    with CONFIG.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print(config["project"]["name"])
    print("Version :", config["project"]["version"])
    print("=" * 50)


if __name__ == "__main__":
    main()