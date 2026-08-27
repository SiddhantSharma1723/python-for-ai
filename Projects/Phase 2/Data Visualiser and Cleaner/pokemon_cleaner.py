import pandas as pd
import matplotlib.pyplot as plt

def clean_data(path):
        
    pokemon = pd.read_csv(path)

    pokemon["HP"] = pd.to_numeric(pokemon["HP"],errors = "coerce")

    pokemon["Type2"] = pokemon["Type2"].replace({
        "None": "Unknown"
    })
    pokemon["Type2"] = pokemon["Type2"].fillna("Unknown")
    pokemon["Type1"] = pokemon["Type1"].astype(str).str.strip().str.title()

    pokemon = pokemon.drop_duplicates()

    q1 = pokemon["Attack"].quantile(0.25)
    q3 = pokemon["Attack"].quantile(0.75)

    IQR = q3 - q1

    lower = q1 - 1.5*IQR
    upper = q3 + 1.5*IQR
    pokemon = pokemon[(pokemon["Attack"]>=lower) & (pokemon["Attack"]<=upper)]

    return pokemon

def make_plots(pokemon):

    attack_distribution = pokemon["Attack"].plot(kind = "hist",bins = 20)
    plt.savefig("attack_distribution.png")
    plt.close()

    pokemon.boxplot(column = "Defense",by = "Type1")
    plt.savefig("Defense by Type1.png")
    plt.close()

    pokemon.boxplot(column = "Defense",by = "Legendary")
    plt.savefig("Defense by Legendary.png")
    plt.close()

    pd.plotting.scatter_matrix(pokemon[["Attack","Defense"]])
    plt.savefig("Attack vs Defense.png")
    plt.close()

    pd.plotting.scatter_matrix(pokemon[["Speed","Sp_Atk"]])
    plt.savefig("Speed vs Sp_Atk.png")
    plt.close()

if __name__ == "__main__":
    pokemon = clean_data("pokemon_stats_messy.csv")
    pokemon.to_csv("pokemon_stats_clean.csv",index = False)
    make_plots(pokemon)
    print(f"Cleaned {len(pokemon)} rows. Saved the file as pokemon_stats_clean.csv and 5 plots as png.")
