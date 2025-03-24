#ifndef _FRAMEWORK_
#define _FRAMEWORK_
#include <map>
#include <string>
#include <fstream>
#include "ValueItem.H"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;


using namespace std;

typedef map<int,double> DBLMAP;
typedef map<int,double>::iterator DBLMAP_ITER;
typedef vector<double> DBLVECT;


class valManager;
class SubGraphMgr;
class GOMgr;
class GeneNameManager;
class Framework
{
	public:
	Framework();
	~Framework();
	int init(const char* subgraphFName, const char* genelist, const char* gosuff);
	int readBackgroundDist(const char*);
	int start(double,const char*,const char*);
	py::list start_py(py::object, py::str);
	py::list runEnrichAnalyzer(py::dict, py::str, py::list, py::str, py::object, py::str);
	int getTermEnrichments(double,double,double,int,const char*);
	int estimateQvalues(double,ofstream&);
	py::list estimateQvalues_ToPyDict(double);
	int estimateQvalues_PerSG(double,ofstream&);
	py::list estimateQvalues_PerSG_ToPyDict(double);
	int findEnrichmentHits(SubGraph*,double,int&);
	int showGraph(SubGraph*,int,double,ofstream&);
	int findTermEnrichmentHits(SubGraphMgr* aMgr, double pval, int sgSize, int& hitCnt, int& totalTerms,map<string,int>&,bool);
	int findTermEnrichmentHits(SubGraphMgr* aMgr, double pval, int sgSize, int& hitCnt, int& totalTerms,map<string,int>&,map<int,map<string,int>*>&, bool);
	int getRandomTerms(int,double);
	double getAvgRandHitCnt(double,int);
	int filter(double,ofstream&,int);
	double getFunctionalDist(SubGraph* sg1, SubGraph* sg2);
	int genBackgroundDistForK(int graphSize);
	int getDistPerTerm(int graphSize);
	int getEnrichments(map<string,double>& enrichPvals, map<string,double>& depletePvals, map<string,int>& geneList, int gid, map<string,string>& ttMap);
	int clearDists();
	double getCorrectedPval(int,double);
	int getGenesInTerm(string&,map<string,int>&);
	int getTotalAnnotatedGenes(map<string,int>&);
	SubGraphMgr sgMgr;
	map<int,map<string,DBLMAP*>*> allRandDists;
	map<string,DBLVECT*> randPvals;
	map<string,DBLMAP*> randDists;
	map<string,int> randomTerms;
	map<int,string> randomTermIDs;
	GOMgr goMgr;
	GeneNameMapper gnm;
	map<int,DBLMAP*> globalRandDist;
	map<string,int> genesInTerm;
};
#endif // _FRAMEWORK_
