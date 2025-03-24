#ifndef _GOTERM_MGR
#define _GOTERM_MGR

#include <map>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace std;

class GOTerm;
class GOMgr
{
	public:
		GOMgr();
		~GOMgr();
		int readGOs(const char*);
		int readGOs(py::str);
		int readTermHierarchy(const char*);
		map<string,GOTerm*>& getAllGOs();
		GOTerm* getGO(const char*);
		int getGOForGene(const char*,map<string,string>&,map<string,string>&);
		map<string,string>* getGOForGene(const char*);
		int getBackgroundDist(map<string,string>&,map<string,double>&);
		int getBackgroundDist(map<string,double>&);
		map<string,map<string,string>*>& getGOForGenes();
		int getGeneCnt();
		int getGeneCntForTerm(const char*);
		int getRootDistForTerm(const char*);
	private:
		int removeDashes(const char*, char*);
		int populateGOData(char*);
		map<string,GOTerm*> gotermSet;
		map<string,map<string,string>*> geneGOSet;

		map<string,int> geneSet;
		map<string,int> termRootLevel;
		int gotermCnt;
};
#endif
