#ifndef _ENRICHANALYZER_
#define _ENRICHANALYZER_

#include <map>
#include <string>
#include <fstream>
#include "subGraphMgr.H"
#include "GOMgr.H"
#include "GeneNameMapper.H"

using namespace std;

typedef map<int,double> DBLMAP;
typedef map<int,double>::iterator DBLMAP_ITER;
typedef vector<double> DBLVECT;

class SubGraphMgr;
class GOMgr;
class GeneNameManager;

class enrichAnalyzer
{
    	public:
    	enrichAnalyzer();
    	~enrichAnalyzer();
    	int init();
    	int start();
    	int getEnrichment();

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


}


    gnm.initGeneNames();
    GoMgr.initGO()

#endif