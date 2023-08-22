from collections import namedtuple
from rdkit import Chem
from rdkit.Chem.PandasTools import LoadSDF
import os

def merge(mol, marked, aset):
  bset = set()
  for idx in aset:
    atom = mol.GetAtomWithIdx(idx)
    for nbr in atom.GetNeighbors():
      jdx = nbr.GetIdx()
      if jdx in marked:
        marked.remove(jdx)
        bset.add(jdx)
  if not bset:
    return
  merge(mol, marked, bset)
  aset.update(bset)


# atoms connected by non-aromatic double or triple bond to any heteroatom
# c=O should not match (see fig1, box 15).  I think using A instead of * should sort that out?
PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
# atoms in non aromatic carbon-carbon double or triple bonds
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
# acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
# all atoms in oxirane, aziridine and thiirane rings
PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')

PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)


def identify_functional_groups(mol):
  marked = set()
  #mark all heteroatoms in a molecule, including halogens
  for atom in mol.GetAtoms():
    if atom.GetAtomicNum() not in (6, 1):  # would we ever have hydrogen?
      marked.add(atom.GetIdx())

#mark the four specific types of carbon atom
  for patt in PATT_TUPLE:
    for path in mol.GetSubstructMatches(patt):
      for atomindex in path:
        marked.add(atomindex)

#merge all connected marked atoms to a single FG
  groups = []
  while marked:
    grp = set([marked.pop()])
    merge(mol, marked, grp)
    groups.append(grp)


#extract also connected unmarked carbon atoms
  ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type'])
  ifgs = []
  for g in groups:
    uca = set()
    for atomidx in g:
      for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
        if n.GetAtomicNum() == 6:
          uca.add(n.GetIdx())
    ifgs.append(
      ifg(atomIds=tuple(list(g)), atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True),
          type=Chem.MolFragmentToSmiles(mol, g.union(uca), canonical=True)))
  return ifgs


def main():
    data_path = 'data/'
    dataset_path = 'MPNN-QM9/'
    sdf_name = 'raw/gdb9.sdf'
    sdf_path = os.path.join(data_path, dataset_path, sdf_name)
    fgs_filename = os.path.join(data_path, dataset_path, 'fgs.csv')

    df = LoadSDF(sdf_path, smilesName='SMILES')
    print('sdf loaded')

    fgs_list = []

    df['ifg'] = df['SMILES'].apply(lambda x: identify_functional_groups(Chem.MolFromSmiles(x)))
    print('functional groups identified')

    for ifg in df['ifg']:
        if ifg == None:
            fgs_list.append([None])
        else:
            tmp = ()
            for i in ifg:
                tmp += i.atomIds
            fgs_list.append(list(set(tmp)))
    df['fgs'] = fgs_list
    print('atom index extracted')

    df.to_csv(fgs_filename, index=False)
    print('csv saved: ', fgs_filename)

if __name__ == "__main__":
  main()

