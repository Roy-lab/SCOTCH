#include <iostream>
#include <fstream>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "SubGraph.H"
#include "SubGraphMgr.H"
#include "Distance.H"
#include "GOTerm.H"
#include "GOMgr.H"
#include "GeneNameMapper.H"
#include "HyperGeomPval.H"
#include "Framework.H"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;


PYBIND11_MODULE(pyEnrichAnalyzer, m) {
  m.doc() = "pybind11 EnrichAnalyzer Module"; // optional module docstring

  py::class_<Framework>(m, "Framework")
    .def(py::init<>()) // exposes the default constructor
    .def("runEnrichAnalyzer", &Framework::runEnrichAnalyzer);
}


int sortfunc(const void* first, const void* second);
int* sortingind=NULL;
double* sortedpvals=NULL;

Framework::Framework()
{
}

Framework::~Framework()
{
}

int 
Framework::init(const char* subgraphFName, const char* geneList, const char* gosuff)
{
	sgMgr.readSubGraphs(subgraphFName);
	//sgMgr.readSubGraphAttributes(attributeFName);
	sgMgr.readGeneList(geneList);
	//gnm.readGeneNames();
	char goFName[1024];
	if((strstr(gosuff,"slim")==NULL) && (strstr(gosuff,"met")==NULL) && (strstr(gosuff,"regnet")==NULL) && (strstr(gosuff,"instance")==NULL))
	{
		sprintf(goFName,"%sgotermap.txt",gosuff);
		goMgr.readGOs(goFName);
		sprintf(goFName,"%sgenecnt.txt",gosuff);
		goMgr.readTermHierarchy(goFName);
	}
	else
	{
		goMgr.readGOs(gosuff);
	}
	return 0;
}

int
Framework::readBackgroundDist(const char* bgDistFName)
{
	ifstream inFile(bgDistFName);
	char* aBuffer=NULL;
	int maxLen=0;
	string strBuff;
	int lineCnt=0;
	while(inFile.good())
	{
		getline(inFile,strBuff);
		if(lineCnt==0)
		{
			lineCnt++;
			continue;
		}
		if(strBuff.length()<=0)
		{
			continue;
		}
		if(maxLen<=strBuff.length())
		{
			maxLen=strBuff.length()+1;
			if(aBuffer!=NULL)
			{
				delete[] aBuffer;
			}
			aBuffer=new char[maxLen];
		}
		strcpy(aBuffer,strBuff.c_str());
		char* tok=strtok(aBuffer," ");
		int tokCnt=0;
		int sgSize=-1;
		string goterm;
		DBLMAP* pDist=NULL;
		while(tok!=NULL)
		{
			if(tokCnt==0)
			{
				goterm.append(tok);
			}
			else if(tokCnt==1)
			{
				sgSize=atoi(tok);
			}
			else
			{
				if(pDist==NULL)
				{
					pDist=new DBLMAP;
				}
				(*pDist)[tokCnt-2]=atof(tok);
			}
			tok=strtok(NULL," ");
			tokCnt++;
		}
		map<string,DBLMAP*>* randDistForK=NULL;
		if(allRandDists.find(sgSize)==allRandDists.end())
		{
			//cout <<"Reading background for subgraphs of size " << sgSize << endl;
			randDistForK=new map<string,DBLMAP*>;
			allRandDists[sgSize]=randDistForK;
		}
		else
		{
			randDistForK=allRandDists[sgSize];
		}
		(*randDistForK)[goterm]=pDist;
	}
	inFile.close();
	cout <<"Read backgrounds for " << allRandDists.size() << " different subgraphs" << endl;
	return 0;
}

int 
Framework::genBackgroundDistForK(int graphSize)
{
	//This will generate iterCnt number of graphs of size graphSize
	vector<SubGraph*> randSGSet;
	int iterCnt=500;
	if(sgMgr.genRandomGraphs(iterCnt,graphSize,randSGSet)==-1)
	{
		return -1;
	}
	map<string,string> termtermMap;
	for(int i=0;i<randSGSet.size();i++)
	{
		SubGraph* sg=randSGSet[i];
		map<string,double> enrichmentVals;
		map<string,double> depleteVals;
		getEnrichments(enrichmentVals,depleteVals,sg->getVertexList(),i,termtermMap);
		for(map<string,double>::iterator aIter=enrichmentVals.begin();aIter!=enrichmentVals.end();aIter++)
		{
			DBLVECT* termPvals=NULL;
			if(randPvals.find(aIter->first)==randPvals.end())
			{
				termPvals=new DBLVECT;
				randPvals[aIter->first]=termPvals;
			}
			else
			{
				termPvals=randPvals[aIter->first];
			}
			termPvals->push_back(aIter->second);
		}
		enrichmentVals.clear();
		depleteVals.clear();
		if(i>0 && ((i%100)==0))
		{
			cout <<".";
			cout.flush();
		}
	}
	for(int i=0;i<randSGSet.size();i++)
	{
		delete randSGSet[i];
	}
	randSGSet.clear();
	termtermMap.clear();
	return 0;
}


int
Framework::getDistPerTerm(int sgsize)
{
	int histBin=10;
	double interval=1.0/((double) histBin);
	map<string,DBLMAP*>* randDistForK=new map<string,DBLMAP*>;
	allRandDists[sgsize]=randDistForK;
	DBLMAP globalDistForK;
	double globalTotal;
	for(map<string,DBLVECT*>::iterator aIter=randPvals.begin();aIter!=randPvals.end();aIter++)
	{
		DBLMAP histogram;
		DBLVECT* pvals=aIter->second;
		double total=0;
		for(int i=0;i<pvals->size();i++)
		{
			double p=(*pvals)[i];
			int bin=(int)floor(log((*pvals)[i])/log(0.1));
			if(bin<0)
			{
				bin=0;
			}
			if(bin>(histBin-1))
			{
				bin=histBin-1;
			}
			if(histogram.find(bin)==histogram.end())
			{
				histogram[bin]=1;
			}
			else
			{
				histogram[bin]=histogram[bin]+1;
			}
			if(globalDistForK.find(bin)==globalDistForK.end())
			{
				globalDistForK[bin]=1;
			}
			else
			{
				globalDistForK[bin]=globalDistForK[bin]+1;
			}
			total=total+1;
			globalTotal=globalTotal+1;
		}
		for(DBLMAP_ITER dIter=histogram.begin();dIter!=histogram.end();dIter++)
		{
			dIter->second=dIter->second/total;
		}
		double cdf=0;
		DBLMAP* pvals_cdf=new DBLMAP;
		(*randDistForK)[aIter->first]=pvals_cdf;
		for(int i=histBin-1;i>=0;i--)
		{
			if(histogram.find(i)==histogram.end())
			{
				(*pvals_cdf)[i]=cdf;
			}
			else
			{
				cdf=cdf+histogram[i];
				(*pvals_cdf)[i]=cdf;
			}
		}
		histogram.clear();
	}
	DBLMAP* globalDistForK_CDF=new DBLMAP;
	for(DBLMAP_ITER dIter=globalDistForK.begin();dIter!=globalDistForK.end();dIter++)
	{
		dIter->second=dIter->second/globalTotal;
	}
	double cdf=0;
	for(int i=histBin-1;i>=0;i--)
	{
		if(globalDistForK.find(i)==globalDistForK.end())
		{
			(*globalDistForK_CDF)[i]=cdf;
		}
		else
		{
			cdf=cdf+globalDistForK[i];
			(*globalDistForK_CDF)[i]=cdf;
		}
	}
	globalRandDist[sgsize]=globalDistForK_CDF;
	globalDistForK.clear();
	return 0;
}

int
Framework::clearDists()
{
	for(map<string,DBLMAP*>::iterator aIter=randDists.begin();aIter!=randDists.end();aIter++)
	{
		aIter->second->clear();
		delete aIter->second;
	}
	randDists.clear();
	for(map<string,DBLVECT*>::iterator aIter=randPvals.begin();aIter!=randPvals.end();aIter++)
	{
		aIter->second->clear();
		delete aIter->second;
	}
	randPvals.clear();
	return 0;
}

double
Framework::getCorrectedPval(int sgSize,double uncorrPval)
{
	double corrPval=0;
	DBLMAP* pvals=globalRandDist[sgSize];
	int histBin=10;
	int bin=(int)floor(log(uncorrPval)/log(0.1));
	if(bin<0)
	{
		bin=0;
	}
	if(bin>(histBin-1))
	{
		bin=histBin-1;
	}
	corrPval=(*pvals)[bin];
	return corrPval;
}


int 
Framework::getGenesInTerm(string& term, map<string,int>& termGeneList)
{
	int hit=0;
	if(genesInTerm.find(term)!=genesInTerm.end())
	{
		hit=genesInTerm[term];
		return hit;
	}
	map<string,int>& bgList=sgMgr.getGeneList();
	for(map<string,int>::iterator tIter=termGeneList.begin();tIter!=termGeneList.end();tIter++)
	{
		if(bgList.find(tIter->first)==bgList.end())
		{
			continue;
		}
		hit++;
	}
	genesInTerm[term]=hit;
	return hit;
}


int 
Framework::getTotalAnnotatedGenes(map<string,int>& geneList)
{
	int hit=0;
	map<string,map<string,string>*>& allAnnotatedGenes=goMgr.getGOForGenes();
	for(map<string,int>::iterator tIter=geneList.begin();tIter!=geneList.end();tIter++)
	{
		if(allAnnotatedGenes.find(tIter->first)==allAnnotatedGenes.end())
		{
			continue;
		}
		hit++;
	}
	return hit;
}

int
Framework::getEnrichments(map<string,double>& enrichPvals, map<string,double>& depletePvals, map<string,int>& geneList, int gid, map<string,string>& ttMap)
{
	map<string,GOTerm*>& geneGOSet=goMgr.getAllGOs();
	string allgenes;
	int total=getTotalAnnotatedGenes(sgMgr.getGeneList());
	//If there is only one GO annotation category the total number of genes is equal to the number of genes in the bg
	if(geneGOSet.size()==1)
	{
		total=sgMgr.getGeneList().size();
	}
	

	for(map<string,GOTerm*>::iterator gIter=geneGOSet.begin();gIter!=geneGOSet.end();gIter++)
	{
		if((strcmp(gIter->first.c_str(),"JUN")==0) && (geneList.size()>10))
		{
			cout <<"Stop here" << endl;
		}
		HyperGeomPval hgp;
		GOTerm* gterm=gIter->second;
		int n1=getGenesInTerm((string&)gIter->first,gterm->getMemberGenes());
		if(n1<5)
		{
			continue;
		}
		//if((strcmp(gIter->first.c_str(),"CentBrain:9-10")==0)|| (strcmp(gIter->first.c_str(),"AntMidGut:11-12")==0)
		if(strstr(gIter->first.c_str(),"VentEct")!=NULL)
	
		{
			cout <<"Stop here " << endl;
		}
		//int n2=goMgr.getGeneCnt()-n1;
		//int n2=sgMgr.getGeneCnt()-n1;
		int n2=total-n1;
		//int t=geneList.size();
		int t=getTotalAnnotatedGenes(geneList);
		if(geneGOSet.size()==1)
		{
			t=geneList.size();
		}
		int k=0;
		map<string,int> hits;
		for(map<string,int>::iterator sIter=geneList.begin();sIter!=geneList.end();sIter++)
		{
			if(gterm->isMember(sIter->first))
			{
				string hitKey(gnm.getCommonName(sIter->first.c_str()));
				hits[hitKey]=0;
				k++;
			}
			if(gIter!=geneGOSet.begin())
			{
				continue;
			}
			if(sIter!=geneList.begin())
			{
				allgenes.append(",");
			}
			allgenes.append(gnm.getCommonName(sIter->first.c_str()));
		}
		if(k<3)
		{
			continue;
		}
		double enpval=1;
		double deppval=1;
		enpval=hgp.getOverRepPval(t,k,n1,n2);
		deppval=hgp.getUnderRepPval(t,k,n1,n2);
	//	double bgprob=((double)n1)/((double)n2);
	//	BinomPval bp;
	//	enpval=bp.getOverRepPval(k,t,bgprob);
	//	deppval=bp.getUnderRepPval(k,t,bgprob);
		char * buffer=new char[strlen(gIter->first.c_str())+1];
		int id=0;
		while(gIter->first.c_str()[id]!='\0')
		{
			char c=gIter->first.c_str()[id];
			if(c!=' ')
			{
				buffer[id]=c;
			}
			else
			{
				buffer[id]=c;
	//			buffer[id]='_';
			}
			id++;
		}
		buffer[id]='\0';
		string newKey(buffer);
		enrichPvals[newKey]=enpval;
		depletePvals[newKey]=deppval;
		ttMap[newKey]=gIter->first;
		delete[] buffer;
	}

	return 0;
}


int 
Framework::start(double fdrthreshold,const char* oFNameSuff,const char* corrtype)
{
	int sgCnt=0;
	int pairsEval=0;
	vector<SubGraph*>& sgSet=sgMgr.getSubGraphs();
	char oFName[1024];
	sprintf(oFName,"%s_details.txt",oFNameSuff);
	ofstream oFile(oFName);
	int skip=0;
	map<int,int> graphMap;
	for(int g=0;g<sgSet.size();g++)
	{
		SubGraph* sg=sgSet[g];
		int k=sg->getVertexList().size();
		if(k<5)
		//if(k<3)
		{
				skip++;
			continue;
		}
		int truehitCnt=0;
		if(strcmp(sgMgr.getSGName(g).c_str(),"FBgn0015602-0")==0)
		{
	//		cout <<"Stophere" << endl;
		}
		findEnrichmentHits(sg,0.05,truehitCnt);
	}
	if(strcmp(corrtype,"persg")==0)
	{
		estimateQvalues_PerSG(fdrthreshold,oFile);
	}
	else if(strcmp(corrtype,"fullgraph")==0)
	{
		estimateQvalues(fdrthreshold,oFile);
	}	
	oFile.close();
	cout <<"Skipped " << skip << endl;
	return 0;
}

py::list
Framework::start_py(py::object fdrthreshold_py, py::str corrtype_py)
{
	double fdrthreshold=1.0;
	string corrtype=corrtype_py;
	py::list enrichments;

	if (py::isinstance<py::float_>(fdrthreshold_py))
	{
		fdrthreshold=py::cast<double>(fdrthreshold_py);
	}else
	{
		cout <<"Error: fdrthreshold is not a float" << endl;
	}

	int sgCnt=0;
	int pairsEval=0;
	vector<SubGraph*>& sgSet=sgMgr.getSubGraphs();
	int skip=0;
	map<int,int> graphMap;
	for(int g=0;g<sgSet.size();g++)
	{
		SubGraph* sg=sgSet[g];
		int k=sg->getVertexList().size();
		if(k<5)
			//if(k<3)
		{
			skip++;
			continue;
		}
		int truehitCnt=0;
		if(strcmp(sgMgr.getSGName(g).c_str(),"FBgn0015602-0")==0)
		{
			//		cout <<"Stophere" << endl;
		}
		findEnrichmentHits(sg,0.05,truehitCnt);
	}
	if(strcmp(corrtype.c_str(),"persg")==0)
	{
		enrichments = estimateQvalues_PerSG_ToPyDict(fdrthreshold);
	}
	else if(strcmp(corrtype.c_str(),"fullgraph")==0)
	{
		enrichments = estimateQvalues_ToPyDict(fdrthreshold);
	}
	return enrichments;
}

int
Framework::estimateQvalues(double fdr,ofstream& oFile)
{
	vector<SubGraph*> sgSet=sgMgr.getSubGraphs();
	//push all pvalue into a long vector
	vector<double> pvals;
	map<int,map<int,string>*> graphTermMap;
	map<int,int> pvalSubgraphMap;
	for(int s=0;s<sgSet.size();s++)
	{
		SubGraph* sg=sgSet[s];
		map<string,double>& enrichVals=sg->getEnrichments();
		map<int,string>* termIDNameMap=new map<int,string>;
		graphTermMap[s]=termIDNameMap;
		for(map<string,double>::iterator aIter=enrichVals.begin();aIter!=enrichVals.end();aIter++)
		{
			if((strcmp(aIter->first.c_str(),"Ubiq")==0) || (strcmp(aIter->first.c_str(),"Mat:1-3")==0) 
			|| (strcmp(aIter->first.c_str(),"Tested")==0)
			||(strcmp(aIter->first.c_str(),"PoleCell:1-3")==0) || (strcmp(aIter->first.c_str(),"NE")==0)
			)
			{
				continue;
			}
			if(aIter->second>0.05)
			{
				//continue;
			}
			(*termIDNameMap)[pvals.size()]=aIter->first;
			pvalSubgraphMap[pvals.size()]=s;
			pvals.push_back(aIter->second);
		}	
	}
	if(pvals.size()==0)
	{
		return 0;
	}
	sortedpvals=new double[pvals.size()];
	sortingind=new int[pvals.size()];
	for(int t=0;t<pvals.size();t++)
	{
		sortedpvals[t]=pvals[t];
		sortingind[t]=t;
	}
	qsort(sortingind,pvals.size(),sizeof(int),&sortfunc);
	vector<double> corrpvals(pvals.size());
	//using Pouya's code
	double rx=1/((double)(pvals.size()));
	double m=(double)pvals.size();
	int maxk=-1;
	for(int k=0;k<pvals.size();k++)
	{
		int sid=sortingind[k];
		double cpval=(pvals[sid]*m)/((double)(k+1));
		corrpvals[sid]=cpval;
		//cout <<" Pvals " << pvals[sid] << " CorrPval " << cpval << endl;
		if(corrpvals[sid]<=fdr)
		{
		//	cout <<"Updating maxk to " << k << " with fdr " << fdr << endl;
			maxk=k;
		}
	}
	//The AFDR is the approximate FDR or the q-value? It is defined as the minimum fdr at which I
	//would call a test significant
	double minfdr=corrpvals[sortingind[corrpvals.size()-1]];
	for(int l=corrpvals.size()-1;l>=0;l--)
	{
		int sid=sortingind[l];
		if(corrpvals[sid]>minfdr)
		{
			corrpvals[sid]=minfdr;
		}
		else 
		{
			minfdr=corrpvals[sid];
		}
	}
	//Now dump the maxk hypothesis
	int k=0;
	int total=getTotalAnnotatedGenes(sgMgr.getGeneList());
	for(int k=0;k<=maxk;k++)
	{
		int sid=sortingind[k];
		int gid=pvalSubgraphMap[sid];
		map<int,string>* termSet=graphTermMap[gid];
		if(termSet->find(sid)==termSet->end())
		{
			cout <<"Did not find term id " << sid << " for sgid " << gid << endl; 
			exit(0);
		}
		GOTerm* goterm=goMgr.getGO((*termSet)[sid].c_str());
		SubGraph* sg=sgSet[gid];
		map<string,int>& memberGenes=sg->getVertexList();
		map<string,int> goMemberGenes=goterm->getMemberGenes();
		int hitCnt=0;
		int n1=getGenesInTerm((string&)(*termSet)[sid],goMemberGenes);
		int t=getTotalAnnotatedGenes(memberGenes);
		for(map<string,int>::iterator mIter=memberGenes.begin();mIter!=memberGenes.end();mIter++)
		{
			if(goMemberGenes.find(mIter->first)==goMemberGenes.end())
			{
				continue;
			}
			hitCnt++;
		}
		double uncorrPval=sortedpvals[sid];
		double corrPval=corrpvals[sid];
		double f1=(double)hitCnt/(double)t;
		double f2=(double)n1/(double)total;
		double foldenr=f1/f2;
		oFile <<sgMgr.getSGName(gid) << "\t" << (*termSet)[sid].c_str() 
		<< "\t"<< uncorrPval << "\t" << corrPval
		<<"\t" << total
		<<"\t" << n1 
		<< "\t" <<t << "\t" << hitCnt 
		<< "\t" << foldenr << endl;
	}
	pvals.clear();
	delete [] sortedpvals;
	delete[] sortingind;
	return 0;
}

py::list
Framework::estimateQvalues_ToPyDict(double fdr)
{
	py::list enrichments;
	vector<SubGraph*> sgSet=sgMgr.getSubGraphs();
	//push all pvalue into a long vector
	vector<double> pvals;
	map<int,map<int,string>*> graphTermMap;
	map<int,int> pvalSubgraphMap;
	for(int s=0;s<sgSet.size();s++)
	{
		SubGraph* sg=sgSet[s];
		map<string,double>& enrichVals=sg->getEnrichments();
		map<int,string>* termIDNameMap=new map<int,string>;
		graphTermMap[s]=termIDNameMap;
		for(map<string,double>::iterator aIter=enrichVals.begin();aIter!=enrichVals.end();aIter++)
		{
			if((strcmp(aIter->first.c_str(),"Ubiq")==0) || (strcmp(aIter->first.c_str(),"Mat:1-3")==0)
			|| (strcmp(aIter->first.c_str(),"Tested")==0)
			||(strcmp(aIter->first.c_str(),"PoleCell:1-3")==0) || (strcmp(aIter->first.c_str(),"NE")==0)
			)
			{
				continue;
			}
			if(aIter->second>0.05)
			{
				//continue;
			}
			(*termIDNameMap)[pvals.size()]=aIter->first;
			pvalSubgraphMap[pvals.size()]=s;
			pvals.push_back(aIter->second);
		}
	}
	if(pvals.size()==0)
	{
		return enrichments;
	}
	sortedpvals=new double[pvals.size()];
	sortingind=new int[pvals.size()];
	for(int t=0;t<pvals.size();t++)
	{
		sortedpvals[t]=pvals[t];
		sortingind[t]=t;
	}
	qsort(sortingind,pvals.size(),sizeof(int),&sortfunc);
	vector<double> corrpvals(pvals.size());
	//using Pouya's code
	double rx=1/((double)(pvals.size()));
	double m=(double)pvals.size();
	int maxk=-1;
	for(int k=0;k<pvals.size();k++)
	{
		int sid=sortingind[k];
		double cpval=(pvals[sid]*m)/((double)(k+1));
		corrpvals[sid]=cpval;
		//cout <<" Pvals " << pvals[sid] << " CorrPval " << cpval << endl;
		if(corrpvals[sid]<=fdr)
		{
		//	cout <<"Updating maxk to " << k << " with fdr " << fdr << endl;
			maxk=k;
		}
	}
	//The AFDR is the approximate FDR or the q-value? It is defined as the minimum fdr at which I
	//would call a test significant
	double minfdr=corrpvals[sortingind[corrpvals.size()-1]];
	for(int l=corrpvals.size()-1;l>=0;l--)
	{
		int sid=sortingind[l];
		if(corrpvals[sid]>minfdr)
		{
			corrpvals[sid]=minfdr;
		}
		else
		{
			minfdr=corrpvals[sid];
		}
	}
	//Now dump the maxk hypothesis
	int k=0;
	int total=getTotalAnnotatedGenes(sgMgr.getGeneList());
	for(int k=0;k<=maxk;k++)
	{
		int sid=sortingind[k];
		int gid=pvalSubgraphMap[sid];
		map<int,string>* termSet=graphTermMap[gid];
		if(termSet->find(sid)==termSet->end())
		{
			cout <<"Did not find term id " << sid << " for sgid " << gid << endl;
			exit(0);
		}
		GOTerm* goterm=goMgr.getGO((*termSet)[sid].c_str());
		SubGraph* sg=sgSet[gid];
		map<string,int>& memberGenes=sg->getVertexList();
		map<string,int> goMemberGenes=goterm->getMemberGenes();
		int hitCnt=0;
		int n1=getGenesInTerm((string&)(*termSet)[sid],goMemberGenes);
		int t=getTotalAnnotatedGenes(memberGenes);
		for(map<string,int>::iterator mIter=memberGenes.begin();mIter!=memberGenes.end();mIter++)
		{
			if(goMemberGenes.find(mIter->first)==goMemberGenes.end())
			{
				continue;
			}
			hitCnt++;
		}
		double uncorrPval=sortedpvals[sid];
		double corrPval=corrpvals[sid];
		double f1=(double)hitCnt/(double)t;
		double f2=(double)n1/(double)total;
		double foldenr=f1/f2;

		ValueItem v(sgMgr.getSGName(gid),
				(*termSet)[sid],
				uncorrPval,
				corrPval,
				total,
				n1,
				t,
				hitCnt,
				foldenr);

		py::dict d = v.valueItemToPythonDict();
		enrichments.append(d);
		//oFile <<sgMgr.getSGName(gid) << "\t" << (*termSet)[sid].c_str()
		//<< "\t"<< uncorrPval << "\t" << corrPval
		//<<"\t" << total
		//<<"\t" << n1
		//<< "\t" <<t << "\t" << hitCnt
		//<< "\t" << foldenr << endl;
	}
	pvals.clear();
	delete [] sortedpvals;
	delete[] sortingind;
	return enrichments;
}


int
Framework::estimateQvalues_PerSG(double fdr,ofstream& oFile)
{
	vector<SubGraph*> sgSet=sgMgr.getSubGraphs();
	//push all pvalue into a long vector
	for(int s=0;s<sgSet.size();s++)
	{
		vector<double> pvals;
		SubGraph* sg=sgSet[s];
		if(strcmp(sgMgr.getSGName(s).c_str(),"FBgn0015602-0")==0)
		{
			cout <<"Stop here " << endl;
		}
		map<string,int>& memberGenes=sg->getVertexList();
		map<string,double>& enrichVals=sg->getEnrichments();
		map<int,string> termIDNameMap;
		for(map<string,double>::iterator aIter=enrichVals.begin();aIter!=enrichVals.end();aIter++)
		{
			if((strstr(aIter->first.c_str(),"Ubiq")!=NULL) || (strcmp(aIter->first.c_str(),"Mat:1-3")==0) 
			|| (strcmp(aIter->first.c_str(),"Tested")==0)
			||(strcmp(aIter->first.c_str(),"PoleCell:1-3")==0) || (strcmp(aIter->first.c_str(),"NE")==0)
			)
			{
				continue;
			}
			if(aIter->second>0.05)
			{
				//continue;
			}
			pvals.push_back(aIter->second);
			termIDNameMap[pvals.size()-1]=aIter->first;
			//cout << sgMgr.getSGName(s) <<" " << aIter->first <<" " << aIter->second << endl;
		}
		if(pvals.size()==0)
		{
			//cout <<"Did not find enrichments for sg " << s << " membergenes " <<memberGenes.size() << endl;
			continue;
		}
		sortedpvals=new double[pvals.size()];
		sortingind=new int[pvals.size()];
		for(int t=0;t<pvals.size();t++)
		{
			sortedpvals[t]=pvals[t];
			sortingind[t]=t;
		}
		qsort(sortingind,pvals.size(),sizeof(int),&sortfunc);
		
		vector<double> corrpvals(pvals.size());
		//using Pouya's code
		double rx=1/((double)(pvals.size()));
		double m=(double)pvals.size();
		int maxk=-1;
		//int maxk=0;
		for(int k=0;k<pvals.size();k++)
		{
			int sid=sortingind[k];
			double cpval=(pvals[sid]*m)/((double)(k+1));
			cout << "Pval " << pvals[sid] << " Corr " << cpval << endl;
			if(cpval>1)
			{
				//cout <<"Found a corrpval to be >1 at k:" <<  k<< " sid:"<< sid<< " pval:"<< pvals[sid] << " corrpval:"<< cpval << endl;
			}
			corrpvals[sid]=cpval;
			/*if(corrpvals[sid]<=fdr)
			{
				maxk=k;
			}*/
		}
		//The AFDR is the approximate FDR or the q-value? It is defined as the minimum fdr at which I
		//would call a test significant
		double minfdr=corrpvals[sortingind[corrpvals.size()-1]];
		for(int l=corrpvals.size()-1;l>=0;l--)
		{
			int sid=sortingind[l];
			//This is important here. We see above in the for loop that the corrpval can exceed 1. 
			//But we cap the minfdr to 1. It exceeds 1 if the pval=1 and k<m. 
			//But below the thresholding brings those guys back to 1.
			if(corrpvals[sid]>minfdr)
			{
				corrpvals[sid]=minfdr;
			}
			else 
			{
				minfdr=corrpvals[sid];
			}
			//As suggested by SHS and verified by SR: Use the calibrated fdr values for maxk
			if(corrpvals[sid]<=fdr)
			{
				//maxk=k;
				maxk++;
			}
		}
		//Now dump the maxk hypothesis
		int total=getTotalAnnotatedGenes(sgMgr.getGeneList());
		for(int k=0;k<=maxk;k++)
		{
			int sid=sortingind[k];
			if(termIDNameMap.find(sid)==termIDNameMap.end())
			{
				cout <<"Did not find term id " << sid << " for sgid " << s << endl; 
				exit(0);
			}
			string hitString;
			GOTerm* goterm=goMgr.getGO((termIDNameMap)[sid].c_str());
			map<string,int> goMemberGenes=goterm->getMemberGenes();
			int hitCnt=0;
			for(map<string,int>::iterator mIter=memberGenes.begin();mIter!=memberGenes.end();mIter++)
			{
				if(goMemberGenes.find(mIter->first)==goMemberGenes.end())
				{
					continue;
				}
				hitCnt++;
				if(hitString.length()>0)
				{
					hitString.append(";");
				}
				hitString.append(gnm.getCommonName(mIter->first.c_str()));
			}
			int n1=getGenesInTerm((string&)termIDNameMap[sid],goMemberGenes);
			int t=getTotalAnnotatedGenes(memberGenes);
			double uncorrPval=sortedpvals[sid];
			double corrPval=corrpvals[sid];
			double f1=(double)hitCnt/(double)t;
			double f2=(double)n1/(double)total;
			double foldenr=f1/f2;
			oFile <<sgMgr.getSGName(s) << "\t" << termIDNameMap[sid].c_str() 
			<< "\t" << uncorrPval 
			<< "\t" << corrPval
			<< "\t" << total
			<< "\t" << n1
			<< "\t" << t
			<< "\t" << hitCnt  
			<< "\t" << foldenr 
			<< "\t"<< hitString.c_str()<< endl;
		}
		pvals.clear();
		delete [] sortedpvals;
		delete[] sortingind;
	}
	return 0;
}


py::list
Framework::estimateQvalues_PerSG_ToPyDict(double fdr)
{
	vector<SubGraph*> sgSet=sgMgr.getSubGraphs();
	py::list enrichments;
	//push all pvalue into a long vector
	for(int s=0;s<sgSet.size();s++)
	{
		vector<double> pvals;
		SubGraph* sg=sgSet[s];
		if(strcmp(sgMgr.getSGName(s).c_str(),"FBgn0015602-0")==0)
		{
			cout <<"Stop here " << endl;
		}
		map<string,int>& memberGenes=sg->getVertexList();
		map<string,double>& enrichVals=sg->getEnrichments();
		map<int,string> termIDNameMap;
		for(map<string,double>::iterator aIter=enrichVals.begin();aIter!=enrichVals.end();aIter++)
		{
			if((strstr(aIter->first.c_str(),"Ubiq")!=NULL) || (strcmp(aIter->first.c_str(),"Mat:1-3")==0)
			|| (strcmp(aIter->first.c_str(),"Tested")==0)
			||(strcmp(aIter->first.c_str(),"PoleCell:1-3")==0) || (strcmp(aIter->first.c_str(),"NE")==0)
			)
			{
				continue;
			}
			if(aIter->second>0.05)
			{
				//continue;
			}
			pvals.push_back(aIter->second);
			termIDNameMap[pvals.size()-1]=aIter->first;
			//cout << sgMgr.getSGName(s) <<" " << aIter->first <<" " << aIter->second << endl;
		}
		if(pvals.size()==0)
		{
			//cout <<"Did not find enrichments for sg " << s << " membergenes " <<memberGenes.size() << endl;
			continue;
		}
		sortedpvals=new double[pvals.size()];
		sortingind=new int[pvals.size()];
		for(int t=0;t<pvals.size();t++)
		{
			sortedpvals[t]=pvals[t];
			sortingind[t]=t;
		}
		qsort(sortingind,pvals.size(),sizeof(int),&sortfunc);

		vector<double> corrpvals(pvals.size());
		//using Pouya's code
		double rx=1/((double)(pvals.size()));
		double m=(double)pvals.size();
		int maxk=-1;
		//int maxk=0;
		for(int k=0;k<pvals.size();k++)
		{
			int sid=sortingind[k];
			double cpval=(pvals[sid]*m)/((double)(k+1));
			cout << "Pval " << pvals[sid] << " Corr " << cpval << endl;
			if(cpval>1)
			{
				//cout <<"Found a corrpval to be >1 at k:" <<  k<< " sid:"<< sid<< " pval:"<< pvals[sid] << " corrpval:"<< cpval << endl;
			}
			corrpvals[sid]=cpval;
			/*if(corrpvals[sid]<=fdr)
			{
				maxk=k;
			}*/
		}
		//The AFDR is the approximate FDR or the q-value? It is defined as the minimum fdr at which I
		//would call a test significant
		double minfdr=corrpvals[sortingind[corrpvals.size()-1]];
		for(int l=corrpvals.size()-1;l>=0;l--)
		{
			int sid=sortingind[l];
			//This is important here. We see above in the for loop that the corrpval can exceed 1.
			//But we cap the minfdr to 1. It exceeds 1 if the pval=1 and k<m.
			//But below the thresholding brings those guys back to 1.
			if(corrpvals[sid]>minfdr)
			{
				corrpvals[sid]=minfdr;
			}
			else
			{
				minfdr=corrpvals[sid];
			}
			//As suggested by SHS and verified by SR: Use the calibrated fdr values for maxk
			if(corrpvals[sid]<=fdr)
			{
				//maxk=k;
				maxk++;
			}
		}
		//Now dump the maxk hypothesis
		int total=getTotalAnnotatedGenes(sgMgr.getGeneList());
		for(int k=0;k<=maxk;k++)
		{
			int sid=sortingind[k];
			if(termIDNameMap.find(sid)==termIDNameMap.end())
			{
				cout <<"Did not find term id " << sid << " for sgid " << s << endl;
				exit(0);
			}
			string hitString;
			GOTerm* goterm=goMgr.getGO((termIDNameMap)[sid].c_str());
			map<string,int> goMemberGenes=goterm->getMemberGenes();
			int hitCnt=0;
			for(map<string,int>::iterator mIter=memberGenes.begin();mIter!=memberGenes.end();mIter++)
			{
				if(goMemberGenes.find(mIter->first)==goMemberGenes.end())
				{
					continue;
				}
				hitCnt++;
				if(hitString.length()>0)
				{
					hitString.append(";");
				}
				hitString.append(gnm.getCommonName(mIter->first.c_str()));
			}
			int n1=getGenesInTerm((string&)termIDNameMap[sid],goMemberGenes);
			int t=getTotalAnnotatedGenes(memberGenes);
			double uncorrPval=sortedpvals[sid];
			double corrPval=corrpvals[sid];
			double f1=(double)hitCnt/(double)t;
			double f2=(double)n1/(double)total;
			double foldenr=f1/f2;
			ValueItem v(
				sgMgr.getSGName(s),
				termIDNameMap[sid],
				uncorrPval,
				corrPval,
				total,
				n1,
				t,
				hitCnt,
				foldenr,
				hitString);

			py::dict d = v.valueItemToPythonDict();
			enrichments.append(d);

			//oFile <<sgMgr.getSGName(s) << "\t" << termIDNameMap[sid].c_str()
			//<< "\t" << uncorrPval
			//<< "\t" << corrPval
			//<< "\t" << total
			//<< "\t" << n1
			//<< "\t" << t
			//<< "\t" << hitCnt
			//<< "\t" << foldenr
			//<< "\t"<< hitString.c_str()<< endl;
		}
		pvals.clear();
		delete [] sortedpvals;
		delete[] sortingind;
	}
	return enrichments;
}

int 
Framework::getTermEnrichments(double maxpval,double minpval, double divider,int k, const char* outputFName)
{
	int sgCnt=0;
	int pairsEval=0;
	cout <<"Hit\tTotal\tRandHit\tRandTotal\tPval\tFDR" << endl;
	double currpval=maxpval;
	ofstream oFile(outputFName);
	while(currpval>=minpval)
	{
		int truehitCnt=0;
		int termCnt=0;
		map<string,int> trueEnrichment;
		map<int,map<string,int>*> trueGraphEnrichment;
		findTermEnrichmentHits(&sgMgr,currpval,k,truehitCnt,termCnt,trueEnrichment,trueGraphEnrichment,true);
		int randhitCnt=0;
		int randtermCnt=0;
		map<string,int> randEnrichment;
		//findTermEnrichmentHits(&randGraphMgr,currpval,k,randhitCnt,randtermCnt,randEnrichment,true);
		double tgProp=(double)truehitCnt/(double)termCnt;
		double rgProp=(double)randhitCnt/(double)randtermCnt;
		double fdr=rgProp/tgProp;
		cout << truehitCnt << "\t" << termCnt <<"\t" << randhitCnt <<"\t" <<randtermCnt << "\t" << currpval <<"\t" <<fdr << endl;
		currpval=currpval/divider;
		if(currpval<0.00015625)
		{
			oFile << endl <<"TERM ENRICHMENT AT PVALUE " << currpval << endl;
			int nonRandHits=0;
	/*		for(map<string,int>::iterator eIter=trueEnrichment.begin();eIter!=trueEnrichment.end();eIter++)
			{
				if(randEnrichment.find(eIter->first)!=randEnrichment.end())
				{
					continue;
				}
				oFile << eIter->first.c_str() <<"\t"<< eIter->second << endl;
				nonRandHits++;
			}*/
			for(map<int,map<string,int>*>::iterator gIter=trueGraphEnrichment.begin();gIter!=trueGraphEnrichment.end();gIter++)
			{
				map<string,int>* enrichedTerms=gIter->second;
				SubGraph* sg=sgMgr.getGraphAt(gIter->first);
				for(map<string,int>::iterator eIter=enrichedTerms->begin();eIter!=enrichedTerms->end();eIter++)
				{	
					if(randEnrichment.find(eIter->first)!=randEnrichment.end())
					{
						continue;
					}
					GOTerm* goterm=goMgr.getGO(eIter->first.c_str());
					map<string,int>& memberGenes=sg->getVertexList();
					map<string,int> goMemberGenes=goterm->getMemberGenes();
					string hitGeneStr;
					string memberGeneStr;
					for(map<string,int>::iterator mIter=memberGenes.begin();mIter!=memberGenes.end();mIter++)
					{
						if(memberGeneStr.length()>0)
						{
							memberGeneStr.append(", ");
						}
						//memberGeneStr.append(mIter->first.c_str());
						const char* cName=gnm.getCommonName(mIter->first.c_str());
						memberGeneStr.append(cName);
						if(goMemberGenes.find(mIter->first)==goMemberGenes.end())
						{
							continue;
						}
						if(hitGeneStr.length()>0)
						{
							hitGeneStr.append(", ");
						}
						hitGeneStr.append(cName);
					}
					oFile <<"SG"<<gIter->first << "\t" << eIter->first.c_str() <<"\t" << goMemberGenes.size() <<"\t" << hitGeneStr.c_str() <<"\t" << memberGeneStr.c_str() << endl;

				}
			}

			//oFile << "Nonrandom hits " <<  nonRandHits<< " out of " << truehitCnt << endl;
		}
		trueEnrichment.clear();
		for(map<int,map<string,int>*>::iterator aIter=trueGraphEnrichment.begin();aIter!=trueGraphEnrichment.end();aIter++)
		{
			aIter->second->clear();
			delete aIter->second;
		}
		trueGraphEnrichment.clear();
		randEnrichment.clear();
	}
	oFile.close();
	return 0;
}

int
Framework::findEnrichmentHits(SubGraph* sg, double pval, int& hitCnt)
{
	hitCnt=0;
	vector<double>& attributes=sg->getAttributes();
	map<string,string> termtermMap;
	map<string,double>& enrichmentVals=sg->getEnrichments();
	map<string,double> depleteVals;
	getEnrichments(enrichmentVals,depleteVals,sg->getVertexList(),0,termtermMap);
	for(map<string,double>::iterator aIter=enrichmentVals.begin();aIter!=enrichmentVals.end();aIter++)
	{
		if(aIter->second>=pval)
		{
	//		continue;
		}
		hitCnt++;
	}
	depleteVals.clear();
	termtermMap.clear();
	return 0;	
}


int
Framework::showGraph(SubGraph* sg, int sgid, double pval, ofstream& oFile)
{
	map<string,int>& attrList=sgMgr.getAttributeNames();
	map<string,double>& enrichVals=sg->getEnrichments();
	for(map<string,double>::iterator aIter=enrichVals.begin();aIter!=enrichVals.end();aIter++)
	{
		if(aIter->second>=pval)
		{
			continue;
		}
		if((strcmp(aIter->first.c_str(),"molecular_function")==0) || (strcmp(aIter->first.c_str(),"cellular_component")==0) || (strcmp(aIter->first.c_str(),"biological_process")==0))
		{
			continue;
		}
		GOTerm* goterm=goMgr.getGO(aIter->first.c_str());
		map<string,int>& memberGenes=sg->getVertexList();
		map<string,int> goMemberGenes=goterm->getMemberGenes();
		string hitGeneStr;
		string memberGeneStr;
		int hitCnt=0;
		for(map<string,int>::iterator mIter=memberGenes.begin();mIter!=memberGenes.end();mIter++)
		{
			if(memberGeneStr.length()>0)
			{
				memberGeneStr.append(", ");
			}
			//memberGeneStr.append(mIter->first.c_str());
			const char* cName=gnm.getCommonName(mIter->first.c_str());
			memberGeneStr.append(cName);
			if(goMemberGenes.find(mIter->first)==goMemberGenes.end())
			{
				continue;
			}
			if(hitGeneStr.length()>0)
			{
				hitGeneStr.append(", ");
			}
			hitGeneStr.append(cName);
			hitCnt++;
		}
		double uncorrPval=aIter->second;
		double corrPval=getCorrectedPval(memberGenes.size(),uncorrPval);
		//oFile <<"SG"<<sgid<< "\t" << aIter->first.c_str() <<"\t" << goMemberGenes.size() << "\t" << memberGenes.size() << "\t" << hitCnt << "\t"<< uncorrPval << "\t" << corrPval <<"\t" << hitGeneStr.c_str() <<"\t" << memberGeneStr.c_str() << endl;
		oFile <<sgMgr.getSGName(sgid) << "\t" << aIter->first.c_str() <<"\t" << goMemberGenes.size() << "\t" << memberGenes.size() << "\t" << hitCnt << "\t"<< uncorrPval << "\t" << corrPval <<"\t" << hitGeneStr.c_str() <<"\t" << memberGeneStr.c_str() << endl;
	}
	return 0;
}

int
Framework::findTermEnrichmentHits(SubGraphMgr* aMgr, double pval, int sgSize, int& hitCnt, int& totalTerms,map<string,int>& enrichedTerms,bool checkAgainstRandom)
{
	vector<SubGraph*>& sgSet=aMgr->getSubGraphs();
	map<string,int>& attrList=aMgr->getAttributeNames();
	for(map<string,int>::iterator aIter=attrList.begin();aIter!=attrList.end();aIter++)
	{
		if((checkAgainstRandom) &&(randomTerms.find(aIter->first)!=randomTerms.end()))
		{
			continue;
		}
		int gid=0;
		while(gid<sgSet.size())
		{
			SubGraph* sg=sgSet[gid];
			if(sg->getVertexList().size()!=sgSize)
			{
				gid++;
				continue;
			}

			if(sg->isEnriched(aIter->second,pval))
			{
				if(enrichedTerms.find(aIter->first)==enrichedTerms.end())
				{
					enrichedTerms[aIter->first]=1;
				}
				else
				{
					enrichedTerms[aIter->first]=enrichedTerms[aIter->first]+1;
				}
			}
			gid++;
		}
	}
	hitCnt=enrichedTerms.size();
	totalTerms=attrList.size();
	return 0;
}


int
Framework::findTermEnrichmentHits(SubGraphMgr* aMgr, double pval, int sgSize, int& hitCnt, int& totalTerms,map<string,int>& enrichedTerms, map<int,map<string,int>*>& enrichedGraphs, bool checkAgainstRandom)
{
	vector<SubGraph*>& sgSet=aMgr->getSubGraphs();
	map<string,int>& attrList=aMgr->getAttributeNames();
	int gid=0;
	map<string,string> termtermMap;
	while(gid<sgSet.size())
	{
		SubGraph* sg=sgSet[gid];
		if(sg->getVertexList().size()!=sgSize)
		{
			gid++;
			continue;
		}
		map<string,double> enrichmentVals;
		map<string,double> depleteVals;
		getEnrichments(enrichmentVals,depleteVals,sg->getVertexList(),gid,termtermMap);

		for(map<string,double>::iterator aIter=enrichmentVals.begin();aIter!=enrichmentVals.end();aIter++)
		{
			if((checkAgainstRandom) &&(randomTerms.find(aIter->first)!=randomTerms.end()))
			{
				continue;
			}
			//if(sg->isEnriched(aIter->second,pval))
			if(aIter->second<pval)
			{
				if(enrichedTerms.find(aIter->first)==enrichedTerms.end())
				{
					enrichedTerms[aIter->first]=1;
				}
				else
				{
					enrichedTerms[aIter->first]=enrichedTerms[aIter->first]+1;
				}
				map<string,int>* termlist=NULL;
				if(enrichedGraphs.find(gid)==enrichedGraphs.end())
				{
					termlist=new map<string,int>;
					enrichedGraphs[gid]=termlist;
				}
				else
				{
					termlist=enrichedGraphs[gid];
				}
				(*termlist)[aIter->first]=0;
			}
		}
		gid++;
		enrichmentVals.clear();
		depleteVals.clear();
	}
	hitCnt=enrichedTerms.size();
	totalTerms=attrList.size();
	return 0;
}


/*Here we get all the possible gene ontology terms and add terms that have 0.0001 chance
 * of a random subgraph to be enriched */
int
Framework::getRandomTerms(int sgsize,double randPval)
{	
	randomTerms.clear();
	randomTermIDs.clear();
	map<string,int>& attrList=sgMgr.getAttributeNames();
	if(allRandDists.find(sgsize)==allRandDists.end())
	{
		cout <<"No background distribution for " << sgsize << endl;
		exit(0);
	}
	map<string,DBLMAP*>* randDistForK=allRandDists[sgsize];
	map<int,int> geneCntDist;
	map<int,int> levelDist;
	for(map<string,int>::iterator aIter=attrList.begin();aIter!=attrList.end();aIter++)
	{
		map<string,DBLMAP*>::iterator rIter=randDistForK->find(aIter->first);
		if(rIter==randDistForK->end())
		{
			cout <<"No background distribution for term "<< rIter->first << endl;
			exit(0);
		}
		DBLMAP* dMap=rIter->second;
		int maxBin=dMap->rbegin()->first;
		DBLMAP_ITER dIter=dMap->begin();
		double corrPval=0;
		while(dIter!=dMap->end())
		{
			double aPval=pow(0.1,(double)(maxBin-dIter->first));
			corrPval=dIter->second;
			if(aPval>=randPval)
			{
				break;
			}
			dIter++;
		}
		if(corrPval>=1e-4)
		{
			randomTerms[aIter->first]=aIter->second;
			randomTermIDs[aIter->second]=aIter->first;
			int gCnt=goMgr.getGeneCntForTerm(aIter->first.c_str());
			int level=goMgr.getRootDistForTerm(aIter->first.c_str());
			int binCnt=gCnt/10;
			if(geneCntDist.find(binCnt)==geneCntDist.end())
			{
				geneCntDist[binCnt]=1;
			}
			else
			{
				geneCntDist[binCnt]=geneCntDist[binCnt]+1;
			}
			if(levelDist.find(level)==levelDist.end())
			{
				levelDist[level]=1;
			}
			else
			{
				levelDist[level]=levelDist[level]+1;
			}
		}
	}
	//cout <<"Genecnt distribution of random terms" << endl;
	for(map<int,int>::iterator aIter=geneCntDist.begin();aIter!=geneCntDist.end();aIter++)
	{
	//	cout <<aIter->first <<"\t" << (double)aIter->second/(double)randomTerms.size() << endl;
	}
	//cout <<"Termlevel distribution of random terms" << endl;
	for(map<int,int>::iterator aIter=levelDist.begin();aIter!=levelDist.end();aIter++)
	{
	//	cout <<aIter->first <<"\t" << (double)aIter->second/(double)randomTerms.size() << endl;
	}
	return 0;
}

double
Framework::getAvgRandHitCnt(double reqdpval,int sgsize)
{
	double avgRandhits=0;
	if(allRandDists.find(sgsize)==allRandDists.end())
	{
		cout <<"No background distribution for " << sgsize << " generating now " << endl;
		genBackgroundDistForK(sgsize);
		getDistPerTerm(sgsize);
	}
	map<string,DBLMAP*>* randDistForK=allRandDists[sgsize];
	for(map<string,DBLMAP*>::iterator rIter=randDistForK->begin();rIter!=randDistForK->end();rIter++)
	{
		DBLMAP* dMap=rIter->second;
		int maxBin=dMap->rbegin()->first;
		DBLMAP_ITER dIter=dMap->begin();
		double corrPval=0;
		while(dIter!=dMap->end())
		{
			double aPval=pow(0.1,(double)(maxBin-dIter->first));
			corrPval=dIter->second;
			if(aPval>=reqdpval)
			{
				break;
			}
			dIter++;
		}
		avgRandhits=avgRandhits+corrPval;
	}
	return avgRandhits;
}



double 
Framework::getFunctionalDist(SubGraph* sg1, SubGraph* sg2)
{
	vector<double>& dVect1=sg1->getAttributes();
	vector<double>& dVect2=sg2->getAttributes();
	if(dVect1.size()!=dVect2.size())
	{
		cout <<"Subgraphs with unequal attibute vectors " << dVect1.size () << " "<< dVect2.size() << endl;
		exit(0);
	}
	Distance d;
	//double dist=d.computeEuclideanDist(dVect1,dVect2);
	double dist=d.computeCC(dVect1,dVect2);
	return dist;
}


int 
sortfunc(const void* first, const void* second)
{
	int ind1=*((int*)first);	
	int ind2=*((int*)second);
	double pval1=sortedpvals[ind1];
	double pval2=sortedpvals[ind2];
	int compstat=0;
	if(pval1<pval2)
	{
		compstat=-1;
	}
	else if(pval1>pval2)
	{
		compstat=1;
	}
	return compstat;
}

py::list
Framework::runEnrichAnalyzer(py::dict assignment, py::str column, py::list gene_list, py::str GO_fileName,
	py::object fdr, py::str test_type)
{
	py::list enrich_list;
	sgMgr.initSubGraphsFromPython(assignment,column);
	sgMgr.initGeneListFromPython(gene_list);
	goMgr.readGOs(GO_fileName);


	enrich_list = start_py(fdr, test_type);
	return enrich_list;

}


int
main(int argc, const char** argv)
{
	if(argc!=7)
	{
		cout <<"Usage: enrichmentAnalyzer subgraph genelist gosuff fdr outputsuff testtype[persg|fullgraph]" << endl;
		return 0;
	}
	Framework fw;
	fw.init(argv[1],argv[2],argv[3]);
	//fw.readBackgroundDist(argv[6]);
	cout <<"Doing fdr analysis enrichment" << endl;
	fw.start(atof(argv[4]),argv[5],argv[6]);
	return 0;
}

