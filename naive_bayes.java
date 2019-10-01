
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Scanner;
import java.util.TreeMap;


public class naive_bayes
{
    public static void histogramsClassification(int bins,int length,TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>> finHisMap, ArrayList<String> gtexstarr, TreeMap<Integer, TreeMap<Integer, TreeMap<Integer, ArrayList<Double>>>> finMap, TreeMap<Integer, ArrayList<String>> collectionArr, TreeMap<Integer, TreeMap<Integer, ArrayList<Double>>> maxminVal)
    {
        ArrayList<String> collArr = gtexstarr;
        
        //ObjectId and predicted Class
        HashMap<Integer, ArrayList<double[]>> map = new HashMap<Integer, ArrayList<double[]>>();
        
        for(int i=0;i<collArr.size();i++)
        {
            String[] splitArr = new String[length+1];
            splitArr = collArr.get(i).split("\\s+");
            
            HashMap<Integer,Double> pc = new HashMap<Integer,Double>();
            
            // For every class -Iterate over class
            for(int j=0;j<finHisMap.size();j++)
            {
                int classNo = Double.valueOf(finHisMap.keySet().toArray()[j].toString()).intValue();
                double Pxc = 1.0;
                double Pcx = 1.0;
                TreeMap<Integer,ArrayList<Double>> dimmap = maxminVal.get(maxminVal.keySet().toArray()[j]);
                //Iterate over all dimensions
                for(int k=1;k<=length;k++)
                {
                    ArrayList<Double> dimensionList = dimmap.get(k);
                    Collections.sort(dimensionList);
                    double S = dimensionList.get(0);
                    double L = dimensionList.get(dimensionList.size()-1);
                    double G =  (L-S)/(bins-3.0);
                    
                    
                    if(G<0.0001||Double.isInfinite(G)||Double.isNaN(G))
                    {
                        G=0.0001;
                    }
                    
                    int binNo = 0;
                    double start = S-(G/2.0);
                    double end = S+(G/2.0);
                    
                    if(Double.parseDouble(splitArr[k-1])<S-(G/2.0))
                    {
                        
                        binNo = 0;
                    }
                    // else if(dimensionList.get(k)>=S+((bins-3)*G)+(G/2.0))
                    else if(Double.parseDouble(splitArr[k-1])>=(L+(G/2.0)))
                    {
                        
                        binNo = (bins-1);
                    }
                    else if(Double.parseDouble(splitArr[k-1])>=start&&Double.parseDouble(splitArr[k-1])<end){binNo =1;}
                    else
                    {
                        
                        for(double z=2;z<bins-1;z++)
                        {
                            start = S+((z-2.0)*G)+(G/2.0);
                            end = S+((z-1.0)*G)+(G/2.0);
                            
                            if(Double.parseDouble(splitArr[k-1])>=start&&Double.parseDouble(splitArr[k-1])<end)
                            {
                                
                                binNo=Double.valueOf(z).intValue();
                                break;
                            }
                            
                        }
                    }
                    Pxc = Pxc * finHisMap.get(classNo).get(k).get(binNo);
                    
                }
                
                double totalSize = collArr.size();
                double classSize = collectionArr.get(classNo).size();
                Pcx = Pxc*(classSize/totalSize);
                
                // every class and First object
                pc.put((Integer) finMap.keySet().toArray()[j], Pcx);
                
            }
            //loop through pc and calculate sum, then divide each value by sum and find the maximum
            double sum = 0.0;
            
            
            for(int n=0;n<pc.size();n++)
            {
                sum = sum + pc.get((Integer) finMap.keySet().toArray()[n]);
                
            }
            
            for(int n=0;n<pc.size();n++)
            {
                //System.out.println("Sum , Val, N "+ sum +" , "+pc.get(finMap.keySet().toArray()[n])+", "+n+" ,"+finMap.keySet().toArray()[n]);
                
                double val = pc.get(finMap.keySet().toArray()[n])/sum;
                //double val = pc.get(finMap.keySet().toArray()[n]);
                if(Double.isNaN(val)){val = 0.0;}
                if(map.get(i)!= null)
                {
                    
                    ArrayList<double[]> arr = map.get(i);
                    double[] newarrval = new double[2];
                    newarrval[0] = (Integer)pc.keySet().toArray()[n];
                    newarrval[1] = val;
                    arr.add(newarrval);
                    map.put(i, arr);
                }
                else
                {
                    ArrayList<double[]> arr = new ArrayList<double[]>();
                    map.put(i,arr);
                    n=n-1;
                }
            }
        }
        
        //Object id , Tied classes and their probability
        HashMap<Integer,ArrayList<double[]>> predictedClass = new HashMap<Integer,ArrayList<double[]>>();
        //Iterate over objects
        for(int i=0;i<map.size();i++)
        {
            
            ArrayList<double[]> arr = new ArrayList<double[]>();
            arr = map.get(i);
            //Find Max Probability
            double maxVal = 0.0;
            for(int k=0;k<arr.size();k++)
            {
                
                if(arr.get(k)[1]>maxVal)
                {
                    maxVal = (arr.get(k)[1]);
                }
            }
            
            //Add all the classes(tied) with maximum probability to predicted classs
            for(int k=0;k<arr.size();k++)
            {
                if(arr.get(k)[1]==maxVal)
                {
                    if(predictedClass.get(i)!=null)
                    {
                        
                        ArrayList<double[]> arrD = predictedClass.get(i);
                        double[] ar = new double[2];
                        ar[0] = arr.get(k)[0];
                        ar[1] = arr.get(k)[1];
                        arrD.add(ar);
                        predictedClass.put(i, arrD);
                    }
                    else
                    {
                        ArrayList<double[]> arrD = new ArrayList<double[]>();
                        predictedClass.put(i, arrD);
                        k = k-1;
                    }
                }
            }
        }
        
        double totalAccuracy = 0.0;
        double count = 0.0;
        
        
        for(int i=0;i<predictedClass.size();i++)
        {
            int object_id = i;
            int predicted_class = 0;
            double probability = 0.0;
            int true_class = Double.valueOf(Double.parseDouble(collArr.get(i).split("\\s+")[collArr.get(i).split("\\s+").length-1])).intValue();
            
            double accuracy = 0.0;
            //System.out.println("i"+i+","+predictedClass.get(57));
            for(int k=0;k<predictedClass.get(i).size();k++)
            {
                if(k==0)
                {
                    predicted_class =  Double.valueOf(predictedClass.get(i).get(k)[0]).intValue();
                    probability = predictedClass.get(i).get(k)[1];
                    
                }
            }
            
            if(predicted_class==true_class)
            {
                accuracy=1.0/predictedClass.get(i).size();
            }
            
            System.out.printf("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n",
                              object_id, predicted_class, probability, true_class, accuracy);
            
            totalAccuracy = totalAccuracy+accuracy;
            count = count+1;
        }
        System.out.println("");
        System.out.printf("classification accuracy=%6.4f\n", (totalAccuracy/count));
    }
    
    
    public static void histograms(TreeMap<Integer, ArrayList<String>> collectionArr, int length, int bin, ArrayList<String> gtexstarr)
    {   double bins = bin;
        // ClassNo      DimensionNo   Data
        TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>> maxminVal = new TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>>();
        //ClassNo		Dimension No	  arraylist.get(0) -> bin 0 probability ,  arraylist.get(1) -> bin 1 probability and so on..
        TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>> finHisMap = new TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>>();
        //Iterating over class
        for(int i=0;i<collectionArr.size();i++)
        {
            ArrayList<String> list = new ArrayList<String>();
            list = collectionArr.get(collectionArr.keySet().toArray()[i]);
            
            // Iterating through every lines in class
            for(int k=0;k<list.size();k++)
            {
                //TreeMap<Integer,ArrayList<Double>> dimensionMap =  new TreeMap<Integer,ArrayList<Double>>();
                
                String[] splitArr = new String[length];
                
                splitArr = list.get(k).split("\\s+");
                
                //Iterating over dimension
                for(int l=0;l<splitArr.length-1;l++)
                {
                    if(maxminVal.get(collectionArr.keySet().toArray()[i])!=null)
                    {
                        if(maxminVal.get(collectionArr.keySet().toArray()[i]).get(l+1)!=null)
                        {
                            ArrayList<Double> maxminValList = maxminVal.get(collectionArr.keySet().toArray()[i]).get(l+1);
                            maxminValList.add(Double.parseDouble(splitArr[l]));
                            Collections.sort(maxminValList);
                            maxminVal.get(collectionArr.keySet().toArray()[i]).put(l+1,maxminValList);
                        }
                        else
                        {
                            ArrayList<Double> maxminValList = new ArrayList<Double>();
                            maxminValList.add(Double.parseDouble(splitArr[l]));
                            Collections.sort(maxminValList);
                            maxminVal.get(collectionArr.keySet().toArray()[i]).put(l+1,maxminValList);
                        }
                    }
                    else
                    {
                        TreeMap<Integer,ArrayList<Double>> dimensionMap =  new TreeMap<Integer,ArrayList<Double>>();
                        ArrayList<Double> maxminValList = new ArrayList<Double>();
                        maxminValList.add(Double.parseDouble(splitArr[l]));
                        Collections.sort(maxminValList);
                        dimensionMap.put(l+1, maxminValList);
                        maxminVal.put((Integer) collectionArr.keySet().toArray()[i], dimensionMap);
                        
                    }
                }
            }
            
        }
        
        //Class->[Dimemnsion No, Dimension Arr]->[bin no, bin Arr]
        TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>>> finMap = new TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>>>();
        //Iterate over class
        for(int i =0;i<maxminVal.size();i++)
        {
            TreeMap<Integer,ArrayList<Double>> map = maxminVal.get(maxminVal.keySet().toArray()[i]);
            //Iterate over dimensions arraylist
            for(int j=1;j<=map.size();j++)
            {
                ArrayList<Double> dimensionList = map.get(j);
                Collections.sort(dimensionList);
                double S = dimensionList.get(0);
                double L = dimensionList.get(dimensionList.size()-1);
                double G =  (L-S)/(bins-3.0);
                
                
                if(G<0.0001||Double.isInfinite(G)||Double.isNaN(G))
                {
                    G=0.0001;
                }
                
                //Iterate over elements in each dimension
                for(int k=0;k<dimensionList.size();k++)
                {
                    int binNo = 0;
                    double start = S-(G/2.0);
                    double end = S+(G/2.0);
                    
                    if(dimensionList.get(k)<S-(G/2.0))
                    {
                        
                        binNo = 0;
                    }
                    // else if(dimensionList.get(k)>=S+((bins-3)*G)+(G/2.0))
                    else if(dimensionList.get(k)>=(L+(G/2.0)))
                    {
                        
                        binNo =  Double.valueOf(bins-1).intValue();
                    }
                    else if(dimensionList.get(k)>=start&&dimensionList.get(k)<end){binNo =1;}
                    else
                    {
                        
                        for(double z=2;z<bins-1;z++)
                        {
                            start = S+((z-2.0)*G)+(G/2.0);
                            end = S+((z-1.0)*G)+(G/2.0);
                            
                            if(dimensionList.get(k)>=start&&dimensionList.get(k)<end)
                            {
                                
                                binNo= Double.valueOf(z).intValue();
                                break;
                            }
                            
                        }
                    }
                    
                    if(finMap.get(maxminVal.keySet().toArray()[i])!=null)
                    {
                        if(finMap.get(maxminVal.keySet().toArray()[i]).get(j)!=null)
                        {
                            if(finMap.get(maxminVal.keySet().toArray()[i]).get(j).get(binNo)!=null){
                                finMap.get(maxminVal.keySet().toArray()[i]).get(j).get(binNo).add(dimensionList.get(k));}
                            else
                            {
                                ArrayList<Double> lists = new ArrayList<Double>();
                                lists.add(dimensionList.get(k));
                                finMap.get(maxminVal.keySet().toArray()[i]).get(j).put(binNo, lists);
                                
                            }
                        }
                        else
                        {
                            TreeMap<Integer,ArrayList<Double>> tree = new TreeMap<Integer,ArrayList<Double>>();
                            ArrayList<Double> lists = new ArrayList<Double>();
                            tree.put(binNo, lists);
                            finMap.get(maxminVal.keySet().toArray()[i]).put(j, tree);
                            k=k-1;
                        }
                    }
                    else
                    {
                        TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>> mapS = new TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>>();
                        finMap.put((Integer) maxminVal.keySet().toArray()[i], mapS);
                        k=k-1;
                    }
                }
            }
            
        }
        
        // TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>>> finMap
        //iterate over class
        for(int i=0;i<finMap.size();i++)
        {
            int classNo =  Double.valueOf(maxminVal.keySet().toArray()[i].toString()).intValue();
            int classSize = maxminVal.get(maxminVal.keySet().toArray()[i]).get(1).size();
            TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>> dimensonMap =finMap.get(maxminVal.keySet().toArray()[i]);
            //System.out.println(dimensonMap.size());
            //iterate over dimension
            for(int j=0;j<dimensonMap.size();j++)
            {
                
                int dimensionNo =  Double.valueOf(dimensonMap.keySet().toArray()[j].toString()).intValue();
                TreeMap<Integer,ArrayList<Double>> binMap = dimensonMap.get(dimensonMap.keySet().toArray()[j]);
                
                //iterate over bins
                for(int k=0;k<bins;k++)
                {
                    
                    ArrayList<Double> list= binMap.get(k);
                    double sum = 0.0;
                    double size = 0.0;
                    if(list!=null)
                    {
                        for(int l=0;l<list.size();l++)
                        {
                            sum = sum+list.get(l);
                        }
                        size = list.size();
                    }
                    
                    ArrayList<Double> dimList = maxminVal.get(maxminVal.keySet().toArray()[i]).get(dimensonMap.keySet().toArray()[j]);
                    Collections.sort(dimList);
                    double G = (dimList.get(dimList.size()-1)-dimList.get(0))/(bins-3);
                    if(G<0.0001||Double.isInfinite(G)||Double.isNaN(G)){G=0.0001;}
                    
                    double val = size/(classSize*G);
                    
                    if(finHisMap.get(classNo)!=null)
                    {
                        if(finHisMap.get(classNo).get(dimensionNo)!=null)
                        {
                            ArrayList<Double> arr = finHisMap.get(classNo).get(dimensionNo);
                            arr.add(val);
                        }
                        else
                        {
                            TreeMap<Integer, ArrayList<Double>> map = finHisMap.get(classNo);
                            ArrayList<Double> arr = new ArrayList<Double>();
                            arr.add(val);
                            map.put(dimensionNo, arr);
                            
                        }
                    }
                    else
                    {
                        TreeMap<Integer, ArrayList<Double>> map = new TreeMap<Integer, ArrayList<Double>>();
                        finHisMap.put(classNo, map);
                        k=k-1;
                    }
                    
                    if(k!=-1){
                        System.out.printf("Class %d, attribute %d, bin %d, P(bin | class) = %.2f",classNo,dimensionNo-1,k,val);
                        System.out.println("");
                    }}
            }
        }
        System.out.println("");
        histogramsClassification(bin,length, finHisMap,gtexstarr,finMap,collectionArr,maxminVal);
    }
    
    //Mixture of Guassians
    public static void mixtureOfGuassians(TreeMap<Integer, ArrayList<String>> collectionArr, int length, int guassians,ArrayList<String> getTextArr)
    {
        TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>> maxminVal = new TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>>();
        //  M step
        //	Class	        Dimension       Guassian        Value  [0]Mean and [1]SD [2]Weight
        TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,double[]>>> SDMeanMap =new TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,double[]>>>();
        //E- Step
        //  Class         Dimension        Guassian       xj     P(xj)
        TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,HashMap<Double,Double>>>> finMap = new TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,HashMap<Double,Double>>>>();
        
        //Iterating over class
        for(int i=0;i<collectionArr.size();i++)
        {
            ArrayList<String> list = new ArrayList<String>();
            list = collectionArr.get(collectionArr.keySet().toArray()[i]);
            
            // Iterating through every lines in class
            for(int k=0;k<list.size();k++)
            {
                //TreeMap<Integer,ArrayList<Double>> dimensionMap =  new TreeMap<Integer,ArrayList<Double>>();
                
                String[] splitArr = new String[length];
                
                splitArr = list.get(k).split("\\s+");
                
                //Iterating over dimension
                for(int l=0;l<splitArr.length-1;l++)
                {
                    if(maxminVal.get(collectionArr.keySet().toArray()[i])!=null)
                    {
                        if(maxminVal.get(collectionArr.keySet().toArray()[i]).get(l+1)!=null)
                        {
                            ArrayList<Double> maxminValList = maxminVal.get(collectionArr.keySet().toArray()[i]).get(l+1);
                            maxminValList.add(Double.parseDouble(splitArr[l]));
                            Collections.sort(maxminValList);
                            maxminVal.get(collectionArr.keySet().toArray()[i]).put(l+1,maxminValList);
                        }
                        else
                        {
                            ArrayList<Double> maxminValList = new ArrayList<Double>();
                            maxminValList.add(Double.parseDouble(splitArr[l]));
                            maxminVal.get(collectionArr.keySet().toArray()[i]).put(l+1,maxminValList);
                        }
                    }
                    else
                    {
                        TreeMap<Integer,ArrayList<Double>> dimensionMap =  new TreeMap<Integer,ArrayList<Double>>();
                        ArrayList<Double> maxminValList = new ArrayList<Double>();
                        maxminValList.add(Double.parseDouble(splitArr[l]));
                        dimensionMap.put(l+1, maxminValList);
                        maxminVal.put((Integer) collectionArr.keySet().toArray()[i], dimensionMap);
                        
                    }
                }
            }
        }
        
        //E- Step
        //  Class         Dimension        Guassian       xj     P(xj)
        //TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,HashMap<Double,Double>>>> finMap = new TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,HashMap<Double,Double>>>>();
        //Iterate over class
        for(int it=0;it<50;it++)
        {
            //Iterate class
            for(int l =0;l<maxminVal.size();l++)
            {
                TreeMap<Integer,ArrayList<Double>> map = maxminVal.get(maxminVal.keySet().toArray()[l]);
                //Iterate over dimensions arraylist
                for(int z=1;z<=map.size();z++)
                {
                    double SD=0.0;
                    double mean = 0.0;
                    double G=0.0;
                    double weight = 0.0;
                    double S=0.0;
                    double L =0.0;
                    ArrayList<Double> dimensionList = map.get(z);
                    Collections.sort(dimensionList);
                    //Only first iteration takes default values
                    if(it==0)
                    {
                        S = dimensionList.get(0);
                        L = dimensionList.get(dimensionList.size()-1);
                        G =  (L-S)/guassians;
                        weight = 1.0/guassians;
                        SD = 1;
                    }
                    //Iterate over guassians
                    
                    for(int i=1;i<=guassians;i++)
                    {
                        if(it==0)
                        {
                            mean = S+(i-1)*G+(G/2.0);
                        }
                        //Already calculated mean, sd and variance, from second iteration of loop
                        else if(it!=0)
                        {
                            double[] val= SDMeanMap.get(maxminVal.keySet().toArray()[l]).get(z).get(i);
                            mean = val[0];
                            SD = val[1];
                            weight = val[2];
                        }
                        //Iterate over elements in each dimension
                        // E-Step
                        //  Class         Dimension        Guassian       xj     P(xj)
                        //TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,double[][]>>> finMap = new TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,HashMap<Double,Double>>>>();
                        
                        for(int k=0;k<dimensionList.size();k++)
                        {
                            double NiXj = 1.0/(SD*Math.sqrt(2*Math.PI));
                            double pow = Math.pow((dimensionList.get(k)-mean),2)/(2*SD*SD);
                            NiXj =(NiXj * Math.pow(Math.E,-pow));
                            
                            NiXj = NiXj*weight;
                            //Check if class is available
                            if(finMap.get(maxminVal.keySet().toArray()[l])!=null)
                            {
                                //Check if dimension is available
                                if(finMap.get(maxminVal.keySet().toArray()[l]).get(z)!=null)
                                {
                                    //Check if guassian is available
                                    if(finMap.get(maxminVal.keySet().toArray()[l]).get(z).get(i)!=null)
                                    {
                                        HashMap<Double, Double> mapThree = finMap.get(maxminVal.keySet().toArray()[l]).get(z).get(i);
                                        mapThree.put(dimensionList.get(k), NiXj);
                                    }
                                    //Add a guassian
                                    else
                                    {
                                        HashMap<Double, Double> mapThree = new HashMap<Double, Double>();
                                        finMap.get(maxminVal.keySet().toArray()[l]).get(z).put(i, mapThree);
                                        k=k-1;
                                    }
                                    
                                }
                                //Add a dimension
                                else
                                {
                                    TreeMap<Integer,HashMap<Double,Double>> mapTwo = new TreeMap<Integer,HashMap<Double,Double>>();
                                    finMap.get(maxminVal.keySet().toArray()[l]).put(z, mapTwo);
                                    k=k-1;
                                }
                            }
                            //Add class to map
                            else
                            {
                                TreeMap<Integer,TreeMap<Integer,HashMap<Double,Double>>> mapOne = new TreeMap<Integer,TreeMap<Integer,HashMap<Double,Double>>>();
                                finMap.put((Integer) maxminVal.keySet().toArray()[l], mapOne);
                                k=k-1;
                            }
                            
                        }
                        //if(it==0){ mean = mean+G;}
                    }
                }
            }
            
            // To find P(x) and find the final pij
            //Iterate over class
            for(int l =0;l<maxminVal.size();l++)
            {
                TreeMap<Integer, ArrayList<Double>> map = maxminVal.get(maxminVal.keySet().toArray()[l]);
                int classNo =  Double.valueOf(maxminVal.keySet().toArray()[l].toString()).intValue();
                //Iterate over dimensions
                for(int z=1;z<=map.size();z++)
                {
                    int dimensionNo = z;
                    ArrayList<Double> dimensionList = map.get(z);
                    
                    //TO ITERATE OVER EACH ELEMENT IN ArrayList
                    for(int m=0;m<dimensionList.size();m++)
                    {
                        double sum = 0.0;
                        //For each element iterated in Arraylist, find P(xj) from Finmap,whcih will be the sum of that elements N(xj)*w
                        //over all guassians
                        double val = dimensionList.get(m);
                        for(int g=1;g<=guassians;g++)
                        {
                            int guassianNo = g;
                            
                            //sum = sum+(dimensionList.get(g).get(mapOne.keySet().toArray()[m])*weight);
                            sum = sum + finMap.get(classNo).get(dimensionNo).get(guassianNo).get(val);
                        }
                        
                        for(int g=1;g<=guassians;g++)
                        {
                            double valForallG = finMap.get(classNo).get(dimensionNo).get(g).get(val);
                            valForallG = valForallG/sum;
                            finMap.get(classNo).get(dimensionNo).get(g).put(val, valForallG);
                            //System.out.printf("Class %d,Diemsion %d,Guassian %d,Xj %.2f,P(xj) %.2f,pij %.2f",classNo,dimensionNo,g,val,sum,valForallG);
                            //System.out.println("");
                        }
                    }
                }
            }
            
            //M Step:
            //Calculating mean, sd and weight
            //		Class	        Dimension       Guassian        Value  [0]Mean and [1]SD [2]Weight
            //TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,HashMap<Double,double[]>>>> SDMeanMap =new TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,HashMap<Double,double[]>>>>();
            //Iterate over class
            for(int l=0;l<maxminVal.size();l++)
            {
                TreeMap<Integer, ArrayList<Double>> map = maxminVal.get(maxminVal.keySet().toArray()[l]);
                int classNo = Double.valueOf(maxminVal.keySet().toArray()[l].toString()).intValue();
                //Iterate over dimensions
                for(int z=1;z<=map.size();z++)
                {
                    int dimensionNo = z;
                    ArrayList<Double> dimensionList = map.get(z);
                    // C1,D1 , C2,D2 and so on
                    //TO ITERATE OVER EACH ELEMENT IN ArrayList dimension wise and get its corresponding pij
                    for(int i=1;i<=guassians;i++)
                    {
                        double sum = 0.0;
                        double sumMul = 0.0;
                        double sumMulsd = 0.0;
                        double sumSd = 0.0;
                        for(int m=0;m<dimensionList.size();m++)
                        {
                            sum = sum+finMap.get(classNo).get(dimensionNo).get(i).get(dimensionList.get(m));
                            sumMul = (dimensionList.get(m)*finMap.get(classNo).get(dimensionNo).get(i).get(dimensionList.get(m)))+sumMul;
                        }
                        //Final Mean
                        double mean = sumMul/sum;
                        
                        for(int m=0;m<dimensionList.size();m++)
                        {
                            sumSd = sum+finMap.get(classNo).get(dimensionNo).get(i).get(dimensionList.get(m));
                            sumMulsd = finMap.get(classNo).get(dimensionNo).get(i).get(dimensionList.get(m))*(Math.pow((dimensionList.get(m)-mean), 2))+sumMulsd;
                            
                        }
                        //Final SD
                        double sd = sumMulsd/sumSd;
                        sd = Math.sqrt(sd);
                        if(sd<0.01)
                        {sd=0.01;
                        }
                        double sumOfPijAllG = 0.0;
                        double sumOfPijG = 0.0;
                        //To calculate weight
                        for(int m=0;m<dimensionList.size();m++)
                        {
                            sumOfPijG = sumOfPijG+finMap.get(classNo).get(dimensionNo).get(i).get(dimensionList.get(m));
                            //  Class         Dimension        Guassian       xj     P(xj)
                            //TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,double[][]>>> finMap = new TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,HashMap<Double,Double>>>>();
                            
                            for(int gw=1;gw<=guassians;gw++)
                            {
                                //Sum of Pij for all guassians
                                // System.out.println(" Class NO , M , GW " + classNo +" , "+m + " , " + gw);
                                sumOfPijAllG = sumOfPijAllG+finMap.get(classNo).get(dimensionNo).get(gw).get(dimensionList.get(m));
                            }
                        }
                        //Final weight
                        double weight = sumOfPijG/sumOfPijAllG;
                        //   Class	       Dimension      Guassian       Value  [0]Mean and [1]SD [2]Weight
                        //TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,double[]>>> SDMeanMap =new TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,double[]>>>();
                        
                        if(SDMeanMap.get(classNo)!=null)
                        {
                            if(SDMeanMap.get(classNo).get(dimensionNo)!=null)
                            {
                                //mean, SD, weight
                                double[] val = new double[3];
                                val[0] = mean;
                                val[1] = sd;
                                val[2] = weight;
                                SDMeanMap.get(classNo).get(dimensionNo).put(i, val);
                                
                            }
                            else
                            {
                                TreeMap<Integer,double[]> mapTwo = new TreeMap<Integer,double[]>();
                                SDMeanMap.get(classNo).put(dimensionNo, mapTwo);
                                i=i-1;
                            }
                        }
                        else
                        {
                            TreeMap<Integer,TreeMap<Integer,double[]>> mapOne = new TreeMap<Integer,TreeMap<Integer,double[]>>();
                            SDMeanMap.put(classNo, mapOne);
                            i = i-1;
                        }
                    }
                }
            }
        }
        
        //Print values
        
        //		Class	        Dimension       Guassian   Value  [0]Mean and [1]SD [2]Weight
        //TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,double[]>>>> SDMeanMap =new TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,double[]>>>();
        
        //Iterate over class
        for(int i=0;i<SDMeanMap.size();i++)
        {
            TreeMap<Integer, TreeMap<Integer, double[]>> map = SDMeanMap.get(maxminVal.keySet().toArray()[i]);
            int classNo =  Double.valueOf(maxminVal.keySet().toArray()[i].toString()).intValue();
            
            //Iterate over dimension
            for(int j=1;j<=map.size();j++)
            {
                int dimensionNo = j;
                
                TreeMap<Integer, double[]> mapOne = map.get(j);
                
                //Iterate over Guassians
                for(int k=1;k<=mapOne.size();k++)
                {
                    
                    double[] val = mapOne.get(k);
                    
                    SDMeanMap.get(classNo).get(dimensionNo).put(k, val);
                    System.out.printf("Class %d, attribute %d, Gaussian %d, mean = %.2f, std = %.2f",classNo,dimensionNo-1,k-1,val[0],val[1],val[2]);
                    System.out.println("");
                }
                
            }
        }
        System.out.println("");
        mGuassiansClassification(SDMeanMap,length,guassians,getTextArr,maxminVal);
    }
    
    //Guassian Mixture Classification                   Class            Dimension     Guassian Value  [0]->Mean and [1]->SD [2]->Weight
    public static void mGuassiansClassification(TreeMap<Integer,TreeMap<Integer,TreeMap<Integer,double[]>>> SDMeanMap, int length, int guassians,ArrayList<String> getTextArr, TreeMap<Integer, TreeMap<Integer, ArrayList<Double>>> maxminVal)
    {
        ArrayList<String> collArr = getTextArr;
        
        //ObjectId and predicted Class
        HashMap<Integer, ArrayList<double[]>> map = new HashMap<Integer, ArrayList<double[]>>();
        
        //Iterate over every line / object in the data
        for(int i=0;i<collArr.size();i++)
        {
            String[] splitArr = new String[length+1];
            splitArr = collArr.get(i).split("\\s+");
            
            HashMap<Integer,Double> pc = new HashMap<Integer,Double>();
            
            // For every class -Iterate over class
            for(int j=0;j<SDMeanMap.size();j++)
            {
                double Pxc = 1.0;
                double Pcx = 1.0;
                //get  mean sd values of all dimensions and find N(x) for the first object, which is the first row
                for(int k=0;k<length;k++)
                {
                    double lPcx = 0.0;
                    //Iterate over Guassians
                    for(int g=1;g<=guassians;g++)
                    {
                        double[] val = SDMeanMap.get(SDMeanMap.keySet().toArray()[j]).get(k+1).get(g);
                        double mean = val[0];
                        double sd = val[1];
                        double weight = val[2];
                        
                        double f = (1.0/sd)*(Math.sqrt(2.0*Math.PI));
                        double pow = Math.pow((Double.parseDouble(splitArr[k])-mean),2);
                        pow = pow/(2.0*sd*sd);
                        
                        //System.out.println("Class : "+j+" Dimension: "+(k+1)+"Guassian: "+g+" SD: "+sd+" Mean: "+mean +" Weight: "+weight);
                        double val1 = weight*f*Math.pow(Math.E,-pow);
                        lPcx = 	lPcx + val1 ;
                        //lPcx =  (weight*f*Math.pow(Math.E,-pow));
                    }
                    
                    Pxc = Pxc * lPcx;
                    
                }
                
                //System.out.println(Pxc);
                //calculate P(Ck|x) and Save the predicted class here and save in pc
                double size1 = maxminVal.get(maxminVal.keySet().toArray()[j]).get(1).size();
                double size2 = collArr.size();
                Pcx = Pxc*(size1/size2);
                
                // every class and First object
                pc.put((Integer) SDMeanMap.keySet().toArray()[j], Pcx);
                
            }
            // ********************
            
            //loop through pc and calculate sum, then divide each value by sum and find the maximum
            double sum = 0.0;
            for(int n=0;n<pc.size();n++)
            {
                sum = sum + pc.get((Integer) SDMeanMap.keySet().toArray()[n]);
                
            }
            
            for(int n=0;n<pc.size();n++)
            {
                double val = pc.get(SDMeanMap.keySet().toArray()[n])/sum;
                
                if(map.get(i)!= null)
                {
                    ArrayList<double[]> arr = map.get(i);
                    double[] newarrval = new double[2];
                    newarrval[0] = (Integer)pc.keySet().toArray()[n];
                    newarrval[1] = val;
                    arr.add(newarrval);
                    map.put(i, arr);
                }
                else
                {
                    ArrayList<double[]> arr = new ArrayList<double[]>();
                    map.put(i,arr);
                    n=n-1;
                }
            }
        }
        
        //Object id , Tied classes and their probability
        HashMap<Integer,ArrayList<double[]>> predictedClass = new HashMap<Integer,ArrayList<double[]>>();
        //Iterate over objects
        for(int i=0;i<map.size();i++)
        {
            ArrayList<double[]> arr = new ArrayList<double[]>();
            arr = map.get(i);
            //Find Max Probability
            double maxVal = 0.0;
            for(int k=0;k<arr.size();k++)
            {
                
                if(arr.get(k)[1]>maxVal)
                {
                    maxVal = (arr.get(k)[1]);
                }
            }
            //Add all the classes(tied) with maximum probability to predicted classs
            for(int k=0;k<arr.size();k++)
            {
                if(arr.get(k)[1]==maxVal)
                {
                    if(predictedClass.get(i)!=null)
                    {
                        ArrayList<double[]> arrD = predictedClass.get(i);
                        double[] ar = new double[2];
                        ar[0] = arr.get(k)[0];
                        ar[1] = arr.get(k)[1];
                        arrD.add(ar);
                        predictedClass.put(i, arrD);
                    }
                    else
                    {
                        ArrayList<double[]> arrD = new ArrayList<double[]>();
                        predictedClass.put(i, arrD);
                        k = k-1;
                    }
                }
            }
        }
        
        double totalAccuracy = 0.0;
        double count = 0.0;
        
        for(int i=0;i<predictedClass.size();i++)
        {
            int object_id = i;
            int predicted_class = 0;
            double probability = 0.0;
            int true_class = Double.valueOf((Double.parseDouble(collArr.get(i).split("\\s+")[collArr.get(i).split("\\s+").length-1]))).intValue();
            
            double accuracy = 0.0;
            for(int k=0;k<predictedClass.get(i).size();k++)
            {
                if(k==0)
                {
                    predicted_class = Double.valueOf(predictedClass.get(i).get(k)[0]).intValue();
                    probability = predictedClass.get(i).get(k)[1];		
                    
                }
            }
            
            if(predicted_class==true_class)
            {
                accuracy=1.0/predictedClass.get(i).size();
            }
            
            System.out.printf("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n", 
                              object_id, predicted_class, probability, true_class, accuracy);
            
            totalAccuracy = totalAccuracy+accuracy;
            count = count+1;			
        }
        System.out.println("");
        System.out.printf("classification accuracy=%6.4f\n", (totalAccuracy/count));
    } 
    
    
    public static void gaussians(TreeMap<Integer, ArrayList<String>> collectionArr, int length, ArrayList<String> gtexstarr)
    {
        TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>> maxminVal = new TreeMap<Integer,TreeMap<Integer,ArrayList<Double>>>();
        
        //Iterating over class 
        for(int i=0;i<collectionArr.size();i++)
        {
            ArrayList<String> list = new ArrayList<String>();
            list = collectionArr.get(collectionArr.keySet().toArray()[i]);
            
            // Iterating through every lines in class
            for(int k=0;k<list.size();k++)
            {
                //TreeMap<Integer,ArrayList<Double>> dimensionMap =  new TreeMap<Integer,ArrayList<Double>>();
                
                String[] splitArr = new String[length];
                
                splitArr = list.get(k).split("\\s+");
                
                //Iterating over dimension
                for(int l=0;l<splitArr.length-1;l++)
                {
                    if(maxminVal.get(collectionArr.keySet().toArray()[i])!=null)
                    {
                        if(maxminVal.get(collectionArr.keySet().toArray()[i]).get(l+1)!=null)
                        {
                            ArrayList<Double> maxminValList = maxminVal.get(collectionArr.keySet().toArray()[i]).get(l+1);
                            maxminValList.add(Double.parseDouble(splitArr[l]));
                            Collections.sort(maxminValList);
                            maxminVal.get(collectionArr.keySet().toArray()[i]).put(l+1,maxminValList);
                        }
                        else
                        {
                            ArrayList<Double> maxminValList = new ArrayList<Double>();
                            maxminValList.add(Double.parseDouble(splitArr[l]));
                            maxminVal.get(collectionArr.keySet().toArray()[i]).put(l+1,maxminValList);
                        }
                    }
                    else
                    {
                        TreeMap<Integer,ArrayList<Double>> dimensionMap =  new TreeMap<Integer,ArrayList<Double>>();
                        ArrayList<Double> maxminValList = new ArrayList<Double>();
                        maxminValList.add(Double.parseDouble(splitArr[l]));
                        dimensionMap.put(l+1, maxminValList);
                        maxminVal.put((Integer) collectionArr.keySet().toArray()[i], dimensionMap);
                        
                    }
                } 					  
            }
            
        }
        
        //Class->[Dimemnsion No, [Mean, Std] ]
        TreeMap<Integer,TreeMap<Integer,double[]>> finMap = new TreeMap<Integer,TreeMap<Integer,double[]>>();
        //Iterate over class
        for(int i =0;i<maxminVal.size();i++)
        {
            double classSize = maxminVal.get(maxminVal.keySet().toArray()[i]).get(1).size();
            TreeMap<Integer,ArrayList<Double>> map = maxminVal.get(maxminVal.keySet().toArray()[i]);
            //Iterate over dimensions arraylist
            for(int j=1;j<=map.size();j++)
            {
                double sumMean = 0.0;
                double sumStd = 0.0;
                
                ArrayList<Double> dimensionList = map.get(j);
                //Calculate Mean
                for(int k=0;k<dimensionList.size();k++)
                { 
                    sumMean = sumMean+dimensionList.get(k);
                }
                double mean = sumMean/classSize;
                double std =0;
                //Calculate Mean
                for(int k=0;k<dimensionList.size();k++)
                { 
                    sumStd = sumStd+Math.pow((dimensionList.get(k)-mean),2);
                }
                //Calculate SD
                sumStd = (sumStd)*(1/(classSize-1));
                std = Math.sqrt(sumStd);
                
                if(std<Math.sqrt(0.0001))
                {
                    std = Math.sqrt(0.0001);
                }
                
                if(finMap.get(maxminVal.keySet().toArray()[i])!=null)
                {
                    if(finMap.get(maxminVal.keySet().toArray()[i]).get(j)!=null)
                    {
                        finMap.get(maxminVal.keySet().toArray()[i]).get(j)[0] = mean;
                        finMap.get(maxminVal.keySet().toArray()[i]).get(j)[1] = std;
                    }
                    else
                    {
                        TreeMap<Integer,double[]> exMap = finMap.get(maxminVal.keySet().toArray()[i]);
                        double[] data = new double[2];
                        exMap.put(j, data);
                        finMap.put((Integer) maxminVal.keySet().toArray()[i], exMap);
                        j=j-1;
                    }
                }
                else
                {
                    TreeMap<Integer,double[]> exMap = new TreeMap<Integer,double[]>();
                    
                    finMap.put((Integer) maxminVal.keySet().toArray()[i], exMap);
                    j=j-1;					 
                }
                
            }
        }			
        
        //TreeMap<Integer,TreeMap<Integer,double[]>> finMap
        //iterate over class
        for(int i=0;i<finMap.size();i++)
        {
            int classNo =  Double.valueOf((maxminVal.keySet().toArray()[i].toString())).intValue();
            TreeMap<Integer,double[]> dimMap = finMap.get(classNo);
            //Iterate through dimensions
            for(int j=1;j<=dimMap.size();j++)
            {
                double[] arr = dimMap.get(j);
                
                System.out.printf("Class %d, attribute %d, mean = %.2f, std = %.2f",classNo,j-1,arr[0],arr[1]);
                System.out.println("");
            }					
        }	
        System.out.println("");
        guassianClassification(finMap,length,maxminVal.size(),gtexstarr,maxminVal);
    }
    
    //Class->[Dimemnsion No, [Mean, Std] ]
    //TreeMap<Integer,TreeMap<Integer,double[]>> finMap = new TreeMap<Integer,TreeMap<Integer,double[]>>();		
    //Classification
    public static void guassianClassification(TreeMap<Integer,TreeMap<Integer,double[]>> finMap,int length,int noOfClasses,ArrayList<String> gtexstarr, TreeMap<Integer, TreeMap<Integer, ArrayList<Double>>> maxminVal)
    {
        ArrayList<String> collArr = gtexstarr;
        
        //ObjectId and predicted Class
        HashMap<Integer, ArrayList<double[]>> map = new HashMap<Integer, ArrayList<double[]>>();
        
        //Iterate over every line / object in the data
        
        for(int i=0;i<collArr.size();i++)
        {
            String[] splitArr = new String[length+1];
            splitArr = collArr.get(i).split("\\s+");
            
            HashMap<Integer,Double> pc = new HashMap<Integer,Double>();
            // For every class	
            
            for(int j=0;j<noOfClasses;j++)
            {		
                // System.out.println(j);
                double Pxc = 1.0;
                double Pcx = 1.0;
                
                //get  mean sd values of all dimensions and find N(x) for the first object, which is the first row
                for(int k=0;k<length;k++)
                { 
                    
                    double[] val = finMap.get(finMap.keySet().toArray()[j]).get(k+1);
                    double mean = val[0];	
                    double sd = val[1];
                    
                    double f = 1/(sd*Math.sqrt(2*Math.PI));
                    double pow = Math.pow((Double.parseDouble(splitArr[k])-mean),2);
                    pow = pow/(2*sd*sd);			
                    Pxc = Pxc * (f*Math.pow(Math.E,-pow));					
                }
                
                //calculate P(Ck|x) and Save the predicted class here and save in pc				 
                double size1 = maxminVal.get(maxminVal.keySet().toArray()[j]).get(1).size();
                double size2 = collArr.size();
                //Pcx = Pxc * P(C)
                Pcx = Pxc*(size1/size2);
                
                // every class and First object 
                pc.put((Integer) finMap.keySet().toArray()[j], Pcx);
                
            } 
            //loop through pc and calculate sum, then divide each value by sum and find the maximum
            double sum = 0.0;
            
            
            for(int n=0;n<pc.size();n++)
            {
                sum = sum + pc.get((Integer) finMap.keySet().toArray()[n]);
                
            }
            
            for(int n=0;n<pc.size();n++)
            {
                //System.out.println("Sum , Val, N "+ sum +" , "+pc.get(finMap.keySet().toArray()[n])+", "+n+" ,"+finMap.keySet().toArray()[n]);
                
                double val = pc.get(finMap.keySet().toArray()[n])/sum;
                //double val = pc.get(finMap.keySet().toArray()[n]);
                
                if(map.get(i)!= null)
                {
                    ArrayList<double[]> arr = map.get(i);
                    double[] newarrval = new double[2];
                    newarrval[0] = (Integer)pc.keySet().toArray()[n];
                    newarrval[1] = val;
                    arr.add(newarrval);
                    
                    map.put(i, arr);
                }	
                else
                {
                    ArrayList<double[]> arr = new ArrayList<double[]>();
                    map.put(i,arr);
                    n=n-1;
                }
            }			 
        }
        
        //Object id , Tied classes and their probability
        HashMap<Integer,ArrayList<double[]>> predictedClass = new HashMap<Integer,ArrayList<double[]>>();
        //Iterate over objects
        for(int i=0;i<map.size();i++)
        {
            ArrayList<double[]> arr = new ArrayList<double[]>();
            arr = map.get(i);
            //Find Max Probability
            double maxVal = 0.0;
            for(int k=0;k<arr.size();k++)
            {
                if(arr.get(k)[1]>maxVal)
                {
                    maxVal = (arr.get(k)[1]);
                }
            }
            //Add all the classes(tied) with maximum probability to predicted classs
            for(int k=0;k<arr.size();k++)
            {
                if(arr.get(k)[1]==maxVal)
                {
                    if(predictedClass.get(i)!=null)
                    {
                        ArrayList<double[]> arrD = predictedClass.get(i);
                        double[] ar = new double[2];
                        ar[0] = arr.get(k)[0];
                        ar[1] = arr.get(k)[1];
                        arrD.add(ar);
                        predictedClass.put(i, arrD);
                    }
                    else
                    {
                        ArrayList<double[]> arrD = new ArrayList<double[]>();
                        predictedClass.put(i, arrD);
                        k = k-1;
                    }
                }						
            }
        }
        
        double totalAccuracy = 0.0;
        double count = 0.0;
        
        for(int i=0;i<predictedClass.size();i++)
        {
            int object_id = i;
            int predicted_class = 0;
            double probability = 0.0;
            int true_class = Double.valueOf(Double.parseDouble(collArr.get(i).split("\\s+")[collArr.get(i).split("\\s+").length-1])).intValue();
            
            double accuracy = 0.0;
            for(int k=0;k<predictedClass.get(i).size();k++)
            {  
                if(k==0)
                {
                    predicted_class = Double.valueOf(predictedClass.get(i).get(k)[0]).intValue();
                    probability = predictedClass.get(i).get(k)[1];		
                    
                }
            }
            
            if(predicted_class==true_class)
            {
                accuracy=1.0/predictedClass.get(i).size();
            }
            
            System.out.printf("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n", 
                              object_id, predicted_class, probability, true_class, accuracy);
            
            totalAccuracy = totalAccuracy+accuracy;
            count = count+1;			
        }
        System.out.println("");
        System.out.printf("classification accuracy=%6.4f\n", (totalAccuracy/count));
    }
    
    
    public static void main(String[] args) throws IOException
    {
        TreeMap<Integer,ArrayList<String>> collectionArr = new TreeMap<Integer,ArrayList<String>>();
        ArrayList<String> gtexstarr = new ArrayList<String>();
        
        
        
        String trainingPath = args[0];
        String testPath = args[1];
        
        FileReader fileReader = new FileReader(trainingPath);
        BufferedReader bufferReader = new BufferedReader(fileReader);
        int length = 0;
        while(bufferReader.ready())
        {
            
            String firstLine = bufferReader.readLine();
            firstLine = firstLine.trim();
            
            length = firstLine.split("\\s+").length;
            
            String[] splitArr = new String[length];
            splitArr = firstLine.split("\\s+");
            String s =  splitArr[length-1];
            if(collectionArr.get(Integer.parseInt(s))!=null)
            {
                ArrayList<String> list = collectionArr.get(Integer.parseInt(s));
                list.add(firstLine);
                collectionArr.put(Integer.parseInt(s), list);
            }
            else
            {
                ArrayList<String> list = new ArrayList<String>();
                list.add(firstLine);
                collectionArr.put(Integer.parseInt(s), list);
            }
        }
        FileReader fileReader1 = new FileReader(testPath);
        BufferedReader bufferReader1 = new BufferedReader(fileReader1);
			    	
			    	while(bufferReader1.ready())
                    {
                        
                        String firstLine = bufferReader1.readLine();
                        firstLine = firstLine.trim();
                        gtexstarr.add(firstLine);
                    }
        bufferReader1.close();
        System.out.println("");
        if(args[2].equalsIgnoreCase("histograms"))
        {
            int bins = Integer.parseInt(args[3]);
            histograms(collectionArr,length-1,bins ,gtexstarr);	
        }
        else if(args[2].equalsIgnoreCase("gaussians"))
        {
            gaussians(collectionArr,length-1,gtexstarr);
        }
        else if(args[2].equalsIgnoreCase("mixtures"))
        {
            int guassians = Integer.parseInt(args[3]);
            mixtureOfGuassians(collectionArr,length-1,guassians,gtexstarr);
        }
        
        bufferReader.close();
        
    }
}