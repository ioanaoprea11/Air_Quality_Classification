from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from functii import *

tabel_invatare_testare = pd.read_csv("Input/pollution.csv", index_col=0)

nan_replace_t(tabel_invatare_testare)

variabile = list(tabel_invatare_testare)
predictori = variabile[:-1]
tinta = variabile[-1]

# Divizare in train-test
x_train, x_test, y_train, y_test = (
    train_test_split(
        tabel_invatare_testare[predictori],
        tabel_invatare_testare[tinta],
        test_size=0.3))

#Construire model liniar
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(x_train, y_train)
clase = model_lda.classes_
q = len(clase)
#print(q)

# Calcul putere discriminare predictori
n = len(x_train)
m = len(predictori)

g = model_lda.means_ - np.mean(x_train.values, axis=0)
dg = np.diag(model_lda.priors_)
#print(dg)

b = g.T @ dg @ g
t = np.cov(x_train.values, rowvar=False, bias=True)
w = t - b

f = (np.diag(b)/(q-1))/(np.diag(w)/(n-q))

p_value = 1 - sts.f.cdf(f, q - 1, n - q)
tabel_predictori = pd.DataFrame(
    data={
        "Putere discriminare": f,
        "p_values": p_value
    }, index=predictori
)
tabel_predictori.to_csv("Output/Predictori.csv")

# Scoruri discriminante(Variabile discriminante) + puterea lor de discriminare
tabel_discriminatori = putere_disc_z(model_lda.transform(x_train), y_train, dg)
tabel_discriminatori.to_csv("Output/Discriminatori.csv")

nr_discriminatori = min(q - 1, m)

z = model_lda.transform(x_test)

tz = pd.DataFrame(z, x_test.index, ["Z" + str(i + 1) for i in range(nr_discriminatori)])
tz.to_csv("Output/z.csv")

zg = tz.groupby(by=y_test.values).mean().values

# Plots
for i in range(nr_discriminatori):
    plot_distributie(z, y_test, clase, i)

for i in range(nr_discriminatori - 1):
    for j in range(i + 1, nr_discriminatori):
        scatterplot_g(z, zg, y_test, clase, i, j)

# Testare
predictie_lda_test = model_lda.predict(x_test)
metrici_lda = calcul_metrici(y_test, predictie_lda_test, clase)
metrici_lda[0].to_csv("Output/MatC_LDA.csv")
metrici_lda[1].to_csv("Output/Acuratete_LDA.csv", index=False)
salvare_erori(y_test, predictie_lda_test, tinta, x_test.index, "LDA")


# Modelul bayesian
model_b = GaussianNB()
model_b.fit(x_train, y_train)

# Testare
predictie_b_test = model_b.predict(x_test)
metrici_b = calcul_metrici(y_test, predictie_b_test, clase)
metrici_b[0].to_csv("Output/MatC_B.csv")
metrici_b[1].to_csv("Output/Acuratete_B.csv", index=False)
salvare_erori(y_test, predictie_b_test, tinta, x_test.index, "Bayes")

# Aplicare model LDA (Acc LDA > Acc B)

x_apply = pd.read_csv("Input/pollution_apply.csv", index_col=0)
predictie_lda = model_lda.predict(x_apply[predictori])
tabel_predictii = pd.DataFrame(
    data={
        "Predictie LDA": predictie_lda
    }, index=x_apply.index
)
tabel_predictii.to_csv("Output/Predictii.csv")

show()