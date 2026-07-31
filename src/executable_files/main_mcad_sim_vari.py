
from src.modules import data_generating_main_investigation
from src.config.config_mcad_variations import variations



def main():

    # "Winding", "Axial", "Operational Point", "Radial"
    with data_generating_main_investigation.MCADDataGenerator() as data_generator:
        data_generator.start(variations["Winding"])



if __name__ == "__main__":
    main()