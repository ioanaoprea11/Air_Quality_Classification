import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import kdeplot, scatterplot
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from pandas.api.types import is_numeric_dtype
import scipy.stats as sts
from matplotlib import cm
from matplotlib.colors import rgb2hex


def nan_replace_t(t):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if any(t[v].isna()):
            if is_numeric_dtype(t[v]):
                t.fillna({v: t[v].mean()}, inplace=True)
            else:
                t.fillna({v: t[v].mean()}, inplace=True)


def plot_distributie(z, y, clase, k=0):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Distributie in axa discriminanta " + str(k + 1), fontsize=16, color="m")
    q = len(clase)
    rampa = cm.get_cmap("rainbow",q)
    paleta = [rgb2hex(rampa(i)) for i in range(q)]
    kdeplot(x=z[:, k], hue=y, fill=True, ax=ax, warn_singular=False,
            palette=paleta, hue_order=clase)


def show():
    plt.show()


def scatterplot_g(z, zg, y,  clase, k1=0, k2=1):
    fig = plt.figure(figsize=(8, 6))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Plot instante si centrii in axele discriminante", fontsize=16, color="m")
    ax.set_xlabel("z" + str(k1 + 1))
    ax.set_ylabel("z" + str(k2 + 1))
    q = len(clase)
    my_map = cm.get_cmap("rainbow", q)
    culori = [rgb2hex(my_map(i)) for i in range(q)]
    scatterplot(x=z[:,k1],y=z[:,k2],hue=y,hue_order=clase,palette=culori,legend=False,ax=ax)
    scatterplot(x=zg[:,k1],y=zg[:,k2],ax=ax,s=200,marker="s",hue=clase,palette=culori,alpha=0.5)


def calcul_metrici(y, y_, clase):
    c = confusion_matrix(y, y_)
    tabel_c = pd.DataFrame(c, clase, clase)
    tabel_c["Acuratete"] = np.round(np.diag(c) * 100 / np.sum(c, axis=1), 3)
    acuratete_medie = tabel_c["Acuratete"].mean()
    acuratete_globala = np.round(sum(np.diag(c)) * 100 / len(y), 3)
    index_CK = cohen_kappa_score(y, y_)
    acuratete = pd.DataFrame(data={
        "Acuratete globala": [acuratete_globala],
        "Acuratete medie": [acuratete_medie],
        "Index Cohen-Kappa": [index_CK]
    })
    return tabel_c, acuratete


def salvare_erori(y, y_, tinta, nume_instante, model):
    tabel = pd.DataFrame(
        data={
            tinta: y,
            "Predictie": y_
        }, index=nume_instante
    )
    tabel[y != y_].to_csv("Output/err_" + model + ".csv")


def putere_disc_z(z, g, dg):
    n, m = z.shape
    q = dg.shape[0]
    zt = pd.DataFrame(z)
    z_ = zt.groupby(by=g.values).mean().values
    vtz = np.diag((1 / n) * z.T @ z)
    vbz = np.diag(z_.T @ dg @ z_)
    alpha = vbz / vtz
    l = alpha * (n - q) / ((1 - alpha) * (q - 1))
    vwz = vtz-vbz
    l_ = (vbz/(q-1))/(vwz/(n-q))
    p_value = 1 - sts.f.cdf(l, q - 1, n - q)
    putere_disc = pd.DataFrame(
        data={
            "Putere discriminare (lambda)": np.round(l, 5),
            "P Values": np.round(p_value, 5)
        }, index=["Z" + str(i + 1) for i in range(m)]
    )
    return putere_disc