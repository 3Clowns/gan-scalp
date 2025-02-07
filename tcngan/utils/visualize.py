import matplotlib.pyplot as plt

def plot_series(real, fake):
    plt.plot(real, label="Real")
    plt.plot(fake.detach().numpy(), label="Fake")
    plt.legend()
    plt.show()