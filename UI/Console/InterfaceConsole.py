from src.UI.InterfaceUsuarioBase import InterfaceUsuarioBase

class InterfaceConsole(InterfaceUsuarioBase):
    def ler(self, mensagem):
        return input(mensagem + ": ")

    def escrever(self, mensagem):
        print(mensagem)
