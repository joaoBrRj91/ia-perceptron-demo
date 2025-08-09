from src.UI.Console.InterfaceConsole import InterfaceConsole

class InterfaceUsuarioFactory:  
    def criar_interface_usuario(tipo="console"):
        if tipo == "gui":
            raise NotImplementedError
        return InterfaceConsole()