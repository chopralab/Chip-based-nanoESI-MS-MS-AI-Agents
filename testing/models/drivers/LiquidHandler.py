def move(x_position:int, y_position:int) -> dict:
        return {
                "x_position": x_position,
                "y_position": y_position
        }

def invalid_move(x_position:int, z_position:int) -> dict:
        return {
                "x_position": x_position,
                "y_position": z_position
        }

def move_no_signature(x_position:int, y_position:int):
        return {
                "x_position": x_position,
                "y_position": y_position
        }

def move_return_string(x_position:int, y_position:int) -> dict:
        return "x_position: " + str(x_position) + ", y_position: " + str(y_position) 

def aspirate(volume:float) -> dict:
        return {
                "volume": volume
        }

def dispense(volume:float) -> dict:
        return {
                "volume": volume
        }

def reset() -> dict:
        return {
                "reset": True
        }