from pymodbus.server.async_io import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
import asyncio
import logging

# Configure logging
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

# Define Modbus Data Blocks
store = ModbusSlaveContext(
    di=ModbusSequentialDataBlock(0, [0] * 100),  # Discrete Inputs
    co=ModbusSequentialDataBlock(0, [0] * 100),  # Coils
    hr=ModbusSequentialDataBlock(0, [0] * 100),  # Holding Registers
    ir=ModbusSequentialDataBlock(0, [0] * 100)   # Input Registers
)

context = ModbusServerContext(slaves=store, single=True)

# Set up server identity (optional)
identity = ModbusDeviceIdentification()
identity.VendorName = 'My Modbus Server'
identity.ProductCode = 'PMod'
identity.VendorUrl = 'http://example.com'
identity.ProductName = 'Modbus Server Example'
identity.ModelName = 'Modbus Server'
identity.MajorMinorRevision = '1.0'

# Background task to monitor coil state
async def monitor_coil_state():
    previous_state = None
    while True:
        current_state = context[0].getValues(1, 1)[0]  # Get the coil state at address 1
        if current_state != previous_state:
            if current_state == 0:
                log.info("Solar panels are switched OFF.")
            else:
                log.info("Solar panels are switched ON.")
            previous_state = current_state
        await asyncio.sleep(1)  # Check every second (adjust as needed)

# Run the Modbus server and the coil monitoring task
if __name__ == "__main__":
    print("Starting Modbus TCP server...")

    # Create an asyncio event loop
    loop = asyncio.get_event_loop()
    # Schedule the server and the monitoring task
    loop.create_task(monitor_coil_state())
    StartTcpServer(context=context, identity=identity, address=("localhost", 5020))