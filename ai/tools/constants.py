PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

CACTUS = "https://cactus.nci.nih.gov/chemical/structure"

OUTPUT_FORMAT = 'JSON'

PROPERTIES = ['MolecularFormula', 'MolecularWeight', 'CanonicalSMILES', 'IsomericSMILES', 'InChI', 'InChIKey', 'IUPACName', 'Title', 'XLogP', \
'ExactMass', 'MonoisotopicMass', 'TPSA', 'Complexity', 'Charge', 'HBondDonorCount', 'HBondAcceptorCount','RotatableBondCount', \
    'HeavyAtomCount', 'IsotopeAtomCount', 'AtomStereoCount', 'DefinedAtomStereoCount', 'UndefinedAtomStereoCount', 'BondStereoCount', \
        'DefinedBondStereoCount', 'UndefinedBondStereoCount', 'CovalentUnitCount', 'PatentCount', 'PatentFamilyCount', 'LiteratureCount', \
            'Volume3D', 'XStericQuadrupole3D', 'YStericQuadrupole3D', 'ZStericQuadrupole3D', 'FeatureCount3D', 'FeatureAcceptorCount3D', 
            'FeatureDonorCount3D', 'FeatureAnionCount3D', 'FeatureCationCount3D', 'FeatureRingCount3D', 'FeatureHydrophobeCount3D', \
                'ConformerModelRMSD3D', 'EffectiveRotorCount3D', 'ConformerCount3D', 'Fingerprint2D']

if __name__ == "__main__":
    print(PROPERTIES)