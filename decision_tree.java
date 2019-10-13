
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.Set;


public class dtree
{
    //Index , Class
    static HashMap<Integer,Integer> classDetails;
    double pruning_thr;
    static double[][] matrix;
    
    class Tree
    {
        Tree left_child;
        Tree right_child;
        int best_attribute;
        double best_threshold;
        //Class,Distribution
        HashMap<Integer,Double> distribution;
        double gain;
        int nodeId;
    }
    
    public dtree( HashMap<Integer,Integer> classDetails,double pruning_thr,double[][] matrix)
    {
        dtree.classDetails = classDetails;
        this.pruning_thr = pruning_thr;
        dtree.matrix=matrix;
    }
    
    //Class,Distribution
    int i=0;
    public Tree oDLT(ArrayList<Integer> exampleList,ArrayList<Integer> attributes,HashMap<Integer,Double> distribution,double pruning_thr)
    {
        
        if(exampleList.size()<50)
        {
            
            Tree tree = new Tree();
            tree.best_attribute = -1;
            tree.best_threshold = -1;
            tree.gain = -1;
            tree.distribution = distribution;
            return tree;
        }
        else
        {
            
            boolean flag = false;
            int length  = matrix[0].length;
            int oldClass = (int) matrix[exampleList.get(0)][length-1];
            
            for(int s=1;s<exampleList.size();s++)
            {
                if(matrix[exampleList.get(s)][length-1]==oldClass)
                {
                    flag = true;
                    oldClass = (int) matrix[exampleList.get(s)][length-1];
                }
                else
                {
                    
                    flag = false;
                    break;
                }
            }
            
            if(flag == true)
            {
                Tree tree = new Tree();
                tree.best_attribute = -1;
                tree.best_threshold = -1;
                tree.gain = -1;
                for(int i=0;i<classDetails.size();i++)
                {
                    if(oldClass==classDetails.get(i))
                        distribution.put(classDetails.get(i), 1.0);
                    else
                        distribution.put(classDetails.get(i), 0.0);
                }
                tree.distribution = distribution;
                return tree;
            }
            else
            {
                ArrayList<Double> list = chooseAttributesOpt(exampleList,attributes);
                int bestAttribute =  (list.get(0)).intValue();
                double bestThreshold = list.get(1);
                double currGain = list.get(2);
                
                Tree tree = new Tree();
                
                tree.best_attribute = bestAttribute-1;
                tree.best_threshold = bestThreshold;
                tree.gain = currGain;
                
                ArrayList<Integer> examples_left = new ArrayList<Integer>();
                ArrayList<Integer> examples_right = new ArrayList<Integer>();
                
                for(int i=0;i<exampleList.size();i++)
                {
                    if(matrix[exampleList.get(i)][bestAttribute-1]<bestThreshold)
                    {
                        examples_left.add(exampleList.get(i));
                    }
                    else if(matrix[exampleList.get(i)][bestAttribute-1]>=bestThreshold)
                    {
                        examples_right.add(exampleList.get(i));
                    }
                }
                
                HashMap<Integer,Double> newDistribution = distribution(exampleList);
                
                tree.left_child = oDLT(examples_left,attributes,newDistribution,pruning_thr);
                tree.right_child = oDLT(examples_right,attributes,newDistribution,pruning_thr);
                
                return tree;
            }
        }
    }
    
    public Tree rDLT(ArrayList<Integer> exampleList,ArrayList<Integer> attributes,HashMap<Integer,Double> distribution,double pruning_thr)
    {
        
        if(exampleList.size()<50)
        {
            
            Tree tree = new Tree();
            tree.best_attribute = -1;
            tree.best_threshold = -1;
            tree.gain = -1;
            tree.distribution = distribution;
            return tree;
        }
        else
        {
            
            boolean flag = false;
            int length  = matrix[0].length;
            int oldClass = (int) matrix[exampleList.get(0)][length-1];
            
            for(int s=1;s<exampleList.size();s++)
            {
                if(matrix[exampleList.get(s)][length-1]==oldClass)
                {
                    flag = true;
                    oldClass = (int) matrix[exampleList.get(s)][length-1];
                }
                else
                {
                    
                    flag = false;
                    break;
                }
            }
            
            if(flag == true)
            {
                Tree tree = new Tree();
                tree.best_attribute = -1;
                tree.best_threshold = -1;
                tree.gain = -1;
                for(int i=0;i<classDetails.size();i++)
                {
                    if(oldClass==classDetails.get(i))
                        distribution.put(classDetails.get(i), 1.0);
                    else
                        distribution.put(classDetails.get(i), 0.0);
                }
                tree.distribution = distribution;
                return tree;
            }
            else
            {
                ArrayList<Double> list = chooseAttributesRnd(exampleList,attributes);
                int bestAttribute =  (list.get(0)).intValue();
                double bestThreshold = list.get(1);
                double currGain = list.get(2);
                
                Tree tree = new Tree();
                
                tree.best_attribute = bestAttribute-1;
                tree.best_threshold = bestThreshold;
                tree.gain = currGain;
                
                ArrayList<Integer> examples_left = new ArrayList<Integer>();
                ArrayList<Integer> examples_right = new ArrayList<Integer>();
                
                for(int i=0;i<exampleList.size();i++)
                {
                    if(matrix[exampleList.get(i)][bestAttribute-1]<bestThreshold)
                    {
                        examples_left.add(exampleList.get(i));
                    }
                    else if(matrix[exampleList.get(i)][bestAttribute-1]>=bestThreshold)
                    {
                        examples_right.add(exampleList.get(i));
                    }
                }
                
                HashMap<Integer,Double> newDistribution = distribution(exampleList);
                
                tree.left_child = rDLT(examples_left,attributes,newDistribution,pruning_thr);
                tree.right_child = rDLT(examples_right,attributes,newDistribution,pruning_thr);
                
                return tree;
            }
        }
    }
    
    public ArrayList<Double> chooseAttributesOpt(ArrayList<Integer> examples, ArrayList<Integer> attributes)
    {
        
        ArrayList<Double> list = new ArrayList<Double>();
        double max_gain = -1;
        double best_attribute = -1;
        double best_threshold = -1;
        
        //Attribute to values mapping of examples, to find min and max element
        HashMap<Integer,ArrayList<Double>> attr_val_map = new HashMap<Integer,ArrayList<Double>>();
        
        for(int j=1;j<=attributes.size();j++)
        {
            ArrayList<Double> valList = new ArrayList<Double>();
            attr_val_map.put(j, valList);
        }
        
        for(int i=0;i<examples.size();i++)
        {
            for(int j=0;j<attributes.size();j++)
            {
                
                ArrayList<Double> valList = attr_val_map.get(j+1);
                
                double attr_val = matrix[examples.get(i)][j];
                valList.add(attr_val);
                attr_val_map.put(j+1, valList);
            }
        }
        
        
        for(int A=1;A<=attributes.size();A++)
        {
            
            //Get the values of attribute and sort it
            ArrayList<Double> valList = attr_val_map.get(A);
            Collections.sort(valList);
            double L = valList.get(0);
            double M = valList.get(valList.size()-1);
            double gain =0.0;
            for(int K=1;K<=50;K++)
            {
                
                double threshold = L+((K*(M-L))/51);
                
                gain = information_gain(examples,A,threshold);
                
                if(gain>max_gain)
                {
                    list = new ArrayList<Double>();
                    best_attribute = A;
                    best_threshold = threshold;
                    max_gain = gain;
                    list.add(best_attribute);
                    list.add(best_threshold);
                    list.add(max_gain);
                }
                
            }
            
        }
        
        return list;
        
    }
    
    public ArrayList<Double> chooseAttributesRnd(ArrayList<Integer> examples, ArrayList<Integer> attributes)
    {
        
        ArrayList<Double> list = new ArrayList<Double>();
        double max_gain = -1;
        double best_attribute = -1;
        double best_threshold = -1;
        
        //Attribute to values mapping of examples, to find min and max element
        HashMap<Integer,ArrayList<Double>> attr_val_map = new HashMap<Integer,ArrayList<Double>>();
        
        for(int j=1;j<=attributes.size();j++)
        {
            ArrayList<Double> valList = new ArrayList<Double>();
            attr_val_map.put(j, valList);
        }
        
        for(int i=0;i<examples.size();i++)
        {
            for(int j=0;j<attributes.size();j++)
            {
                
                ArrayList<Double> valList = attr_val_map.get(j+1);
                
                double attr_val = matrix[examples.get(i)][j];
                valList.add(attr_val);
                attr_val_map.put(j+1, valList);
            }
        }
        Random rand = new Random();
        int A = attributes.get(rand.nextInt(attributes.size()));
        
        //Get the values of attribute and sort it
        ArrayList<Double> valList = attr_val_map.get(A);
        Collections.sort(valList);
        double L = valList.get(0);
        double M = valList.get(valList.size()-1);
        double gain =0.0;
        for(int K=1;K<=50;K++)
        {
            
            double threshold = L+((K*(M-L))/51);
            
            gain = information_gain(examples,A,threshold);
            
            if(gain>max_gain)
            {
                list = new ArrayList<Double>();
                best_attribute = A;
                best_threshold = threshold;
                max_gain = gain;
                list.add(best_attribute);
                list.add(best_threshold);
                list.add(max_gain);
            }
            
        }
        
        return list;
        
    }
    
    
    public double information_gain(ArrayList<Integer> examples,int A,double threshold)
    {
        ArrayList<Integer> examples_left = new ArrayList<Integer>();
        ArrayList<Integer> examples_right = new ArrayList<Integer>();
        
        for(int i=0;i<examples.size();i++)
        {
            
            if(matrix[examples.get(i)][A-1]<threshold)
            {
                examples_left.add(examples.get(i));
            }
            else if(matrix[examples.get(i)][A-1]>=threshold)
            {
                examples_right.add(examples.get(i));
            }
            
        }
        
        //Split the examples class wise
        
        //Class,Corresponding example array
        HashMap<Integer,ArrayList<Integer>> classExm = new HashMap<Integer,ArrayList<Integer>>();
        
        //Length with class
        int length = matrix[examples.get(0)].length;
        
    	   for(int i=0;i<examples.size();i++)
           {
               int currClass = (int) matrix[examples.get(i)][length-1];
               if(classExm.get(currClass)==null)
               {
                   ArrayList<Integer> exm = new ArrayList<Integer>();
                   exm.add(examples.get(i));
                   classExm.put(currClass, exm);
               }
               else
               {
                   ArrayList<Integer> exm = classExm.get(currClass);
                   exm.add(examples.get(i));
                   classExm.put(currClass, exm);
               }
           }
        
        //Split the examples_left class wise
    	   
        //Class,Corresponding example array
        HashMap<Integer,ArrayList<Integer>> classExm_left = new HashMap<Integer,ArrayList<Integer>>();
        
        
        for(int i=0;i<examples_left.size();i++)
        {
            int currClass = (int) matrix[examples_left.get(i)][length-1];
            if(classExm_left.get(currClass)==null)
            {
                ArrayList<Integer> exm = new ArrayList<Integer>();
                exm.add(examples_left.get(i));
                classExm_left.put(currClass, exm);
            }
            else
            {
                ArrayList<Integer> exm = classExm_left.get(currClass);
                exm.add(examples_left.get(i));
                classExm_left.put(currClass, exm);
            }
        }
        
        //Split the examples_right class wise
        
        //Class,Corresponding example array
        HashMap<Integer,ArrayList<Integer>> classExm_right = new HashMap<Integer,ArrayList<Integer>>();
        
        for(int i=0;i<examples_right.size();i++)
        {
            int currClass = (int) matrix[examples_right.get(i)][length-1];
            if(classExm_right.get(currClass)==null)
            {
                ArrayList<Integer> exm = new ArrayList<Integer>();
                exm.add(examples_right.get(i));
                classExm_right.put(currClass, exm);
            }
            else
            {
                ArrayList<Integer> exm = classExm_right.get(currClass);
                exm.add(examples_right.get(i));
                classExm_right.put(currClass, exm);
            }
        }
    	   
        //Calculate size
        double exampleSize = examples.size();
        double exampleSizeLeft = examples_left.size();
        double exampleSizeRight = examples_right.size();
        
        
        //Initialize Entropy
        double entropy_node = 0.0;
        double entropy_left = 0.0;
        double entropy_right = 0.0;
        
        //Calculate entropy for node N
        
        for(int i=0;i<classDetails.size();i++)
        {
            
            if( classExm.get(classDetails.get(i))!=null){
                double classSize = classExm.get(classDetails.get(i)).size();
                entropy_node = entropy_node-((((double) classSize/(double) exampleSize))*(Math.log(((double) classSize/(double) exampleSize))/Math.log(2.0d)));
                
            }
        }
        
        //Calculate the entropy for class_left
        for(int i=0;i<classDetails.size();i++)
        {
            if( classExm_left.get(classDetails.get(i))!=null){
                double classSize = classExm_left.get(classDetails.get(i)).size();
                entropy_left = entropy_left-((((double) classSize/(double) exampleSizeLeft)*(Math.log((double) classSize/(double) exampleSizeLeft)/Math.log(2.0d))));
            }
            
        }
        
        //Calculate the entropy for class_right
        for(int i=0;i<classDetails.size();i++)
        {
            if( classExm_right.get(classDetails.get(i))!=null){
                double classSize = classExm_right.get(classDetails.get(i)).size();
                entropy_right = entropy_right-((((double) classSize/(double) exampleSizeRight)*(Math.log((double) classSize/(double) exampleSizeRight)/Math.log(2.0d))));
            }
            
        }
        
        
        //Calculating information gain
        double information_gain = entropy_node - ((((double) exampleSizeLeft/(double) exampleSize)*(double) entropy_left)+(((double) exampleSizeRight/(double) exampleSize)*(double) entropy_right));
        return information_gain;
    }
    
    public HashMap<Integer,Double> distribution(ArrayList<Integer> exampleList)
    {   //Class,Distribution
        HashMap<Integer,Double> distribution = new HashMap<Integer,Double>();
        //Class,Corresponding example array
        HashMap<Integer,ArrayList<Integer>> distributionSum = new HashMap<Integer,ArrayList<Integer>>();
        
        //Length with class
        int length = matrix[exampleList.get(0)].length;
        
        for(int i=0;i<classDetails.size();i++)
        {
            ArrayList<Integer> exm = new ArrayList<Integer>();
            distributionSum.put(classDetails.get(i), exm);
        }
        
        
        for(int i=0;i<exampleList.size();i++)
        {
            int currClass = (int) matrix[exampleList.get(i)][length-1];
            
            {
                ArrayList<Integer> exm = distributionSum.get(currClass);
                exm.add(exampleList.get(i));
                distributionSum.put(currClass, exm);
            }
        }
        
        double exampleSize   = exampleList.size();
        
        for(int i=0;i<distributionSum.size();i++)
        {
            if(distributionSum.get(classDetails.get(i))!=null){
                int size = distributionSum.get(classDetails.get(i)).size();
                double distrb = size/exampleSize;
                distribution.put(classDetails.get(i), distrb);  }
            else
            {
                distribution.put(classDetails.get(i), 0.0);
            }
        }
        
        return distribution;
        
    }
    
    //To display the result
    public void displayTrainingResult(Tree tree,int treeId)
    {
        int tree_id = treeId;
        
        Queue<Tree> queue = new LinkedList<Tree>();
        Tree current = tree;
        current.nodeId = 1;
        queue.add(current);
        while(!queue.isEmpty())
        {
            
            current = queue.poll();
            System.out.printf("tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n", tree_id, current.nodeId, current.best_attribute, current.best_threshold, current.gain);
            
            if (current.left_child != null)
            {
                current.left_child.nodeId = current.nodeId*2;
                queue.add(current.left_child);
                
            }
            
            if (current.right_child != null)
            {
                current.right_child.nodeId = (current.nodeId*2)+1;
                queue.add(current.right_child);
            }
            
            
        }
        
    }
    
    //Classification
    //Optimized
    
    public void optimizedRanClassification(Tree tree,ArrayList<Integer> examples_test,double[][] matrix_test,ArrayList<Integer> attributes)
    {
        
        double totalAcc = 0;
        double[][] classificationArr = new double[examples_test.size()][4];
        
        for(int i=0;i<examples_test.size();i++)
        {
            Tree curr = tree;
            
            while(curr.best_attribute!=-1)
            {
                double thr = curr.best_threshold;
                int attr = curr.best_attribute;
                
                if(matrix_test[i][attr]<thr)
                {
                    curr = curr.left_child;
                }
                else
                {
                    curr = curr.right_child;
                }
                
            }
            
            HashMap<Integer,Double> dist = curr.distribution;
            double max = -1.0;
            double maxclass = -1;
            int count=0;
            
            for(int d=0;d<dist.size();d++)
            {
                if(dist.get(classDetails.get(d))>=max)
                {
                    max = dist.get(classDetails.get(d));
                    maxclass = classDetails.get(d);
                }
            }
            
            for(int d=0;d<dist.size();d++)
            {
                if(dist.get(classDetails.get(d))==max)
                {
                    count = count+1;
                }
            }
            int acc = 0;
            
            if(maxclass == matrix_test[i][attributes.size()])
            {
                acc = 1/count;
            }
            else
            {
                acc = 0;
            }
            
            totalAcc = totalAcc+acc;
            
            classificationArr[i][0] = i;
            classificationArr[i][1] = maxclass;
            classificationArr[i][2] = matrix_test[i][attributes.size()];
            classificationArr[i][3] = acc;
        }
        
        
        for(int c=0;c<classificationArr.length;c++)
        {
            System.out.printf("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n", (int)classificationArr[c][0],
                              (int)classificationArr[c][1], (int)classificationArr[c][2], classificationArr[c][3]);
            
        }
        
        System.out.printf("classification accuracy=%6.4f\n", totalAcc/examples_test.size());
    }
    
    
    //Classification
    //Random 3
    public void RanClassification3(ArrayList<Tree> list, ArrayList<Integer> examples_test,double[][] matrix_test,ArrayList<Integer> attributes)
    {
        // treeID, Collection of Distribution for all objects
        HashMap<Integer, HashMap<Integer,HashMap<Integer, Double>>> distributionMap = new HashMap<Integer,HashMap<Integer,HashMap<Integer,Double>>>();
        
        for(int t=0;t<list.size();t++)
        {
            Tree tree = list.get(t);
            HashMap<Integer,HashMap<Integer, Double>> distributionList = new HashMap<Integer,HashMap<Integer, Double>>();
            
            for(int i=0;i<examples_test.size();i++)
            {
                Tree curr = tree;
                
                while(curr.best_attribute!=-1)
                {
                    double thr = curr.best_threshold;
                    int attr = curr.best_attribute;
                    
                    if(matrix_test[i][attr]<thr)
                    {
                        curr = curr.left_child;
                    }
                    else
                    {
                        curr = curr.right_child;
                    }
                    
                }
                distributionList.put(i,curr.distribution)	;
            }
            distributionMap.put(t, distributionList);
        }
        
        HashMap<Integer,HashMap<Integer, Double>> distributionSumList = new HashMap<Integer,HashMap<Integer, Double>>();
        
        for(int i=0;i<examples_test.size();i++)
        {
            
            HashMap<Integer, Double> avdist = new HashMap<Integer, Double>();
            
            for(int c=0;c<classDetails.size();c++)
            {
                double sum = 0;
                for(int s=0;s<list.size();s++)
                {
                    sum = sum+distributionMap.get(s).get(i).get(classDetails.get(c));
                }
                avdist.put(classDetails.get(c), sum);
            }
            distributionSumList.put(i,avdist);
            
        }
        
        
        for(int i=0;i<distributionSumList.size();i++)
        {
            HashMap<Integer, Double> dist = distributionSumList.get(i);
            
            for(int m=0;m<classDetails.size();m++)
            {
                double val = dist.get(classDetails.get(m));
                val = val/list.size();
                dist.put(classDetails.get(m), val);
            }
            
            distributionSumList.put(i,dist);
        }
        
        double totalAcc = 0;
        double[][] classificationArr = new double[examples_test.size()][4];
        
        for(int i=0;i<distributionSumList.size();i++){
            
            HashMap<Integer,Double> dist = distributionSumList.get(i);
            double max = -1.0;
            double maxclass = -1;
            int count=0;
            
            for(int d=0;d<dist.size();d++)
            {
                if(dist.get(classDetails.get(d))>=max)
                {
                    max = dist.get(classDetails.get(d));
                    maxclass = classDetails.get(d);
                }
            }
            
            for(int d=0;d<dist.size();d++)
            {
                if(dist.get(classDetails.get(d))==max)
                {
                    count = count+1;
                }
            }
            int acc = 0;
            
            if(maxclass == matrix_test[i][attributes.size()])
            {
                acc = 1/count;
            }
            else
            {
                acc = 0;
            }
            
            totalAcc = totalAcc+acc;
            
            classificationArr[i][0] = i;
            classificationArr[i][1] = maxclass;
            classificationArr[i][2] = matrix_test[i][attributes.size()];
            classificationArr[i][3] = acc;
        }
        
        
        for(int c=0;c<classificationArr.length;c++)
        {
            System.out.printf("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n", (int)classificationArr[c][0],
                              (int)classificationArr[c][1], (int)classificationArr[c][2], classificationArr[c][3]);
            
        }
        
        System.out.printf("classification accuracy=%6.4f\n", totalAcc/examples_test.size());
        
        
    }
    
    public static void main(String[] args) throws IOException
    {
        
        /*String trainingPath = "/Users/archana/Desktop/MCData/pendigits_training.txt";
         //String testPath = "/Users/archana/Desktop/MCData/pendigits_test.txt";
         //int pruning_thr = 50;
         String option = "forest15";*/
        
        String trainingPath = args[0];
        String testPath =  args[1];
        String option = args[2];
        int  pruning_thr = Integer.parseInt(args[3]);
        
        //Load training data into a matrix
        FileReader fileReader = new FileReader(trainingPath);
        BufferedReader bufferReader = new BufferedReader(fileReader);
        ArrayList<String> examples = new ArrayList<String>();
        HashMap<Integer,Integer> classsCount = new HashMap<Integer,Integer>();
        
        while(bufferReader.ready())
        {	   
            
            String firstLine = bufferReader.readLine();
            firstLine = firstLine.trim();
            examples.add(firstLine);
        }
        int length = examples.get(0).split("\\s+").length;
        
        
        matrix= new double[examples.size()][length];
        
        for(int i=0;i<examples.size();i++)
        {
            String[] arr = examples.get(i).split("\\s+");
            
            for(int j=0;j<length;j++)
            {
                
                matrix[i][j] = Double.parseDouble(arr[j]);
            }
        }
        
        
        
        //Load training data into a matrix
        
        FileReader fileReader_test = new FileReader(testPath);
        BufferedReader bufferReader_test = new BufferedReader(fileReader_test);
        ArrayList<String> examples_test = new ArrayList<String>();
        
        
        while(bufferReader_test.ready())
        {	   
            
            String firstLine = bufferReader_test.readLine();
            firstLine = firstLine.trim();
            examples_test.add(firstLine);
        }
        
        
        double[][] matrix_test= new double[examples_test.size()][length];
        
        for(int i=0;i<examples_test.size();i++)
        {
            String[] arr = examples_test.get(i).split("\\s+");
            
            for(int j=0;j<length;j++)
            {
                
                matrix_test[i][j] = Double.parseDouble(arr[j]);
            }
        }
        
        
        
        for(int i=0;i<examples.size();i++)
        {
            int currClass = Integer.parseInt(examples.get(i).split("\\s+")[length-1]);
            if(classsCount.containsKey(currClass))
            {
                int currCount = classsCount.get(currClass);
                currCount = currCount+1;
                classsCount.put(currClass, currCount);
            }
            else
            {
                classsCount.put(currClass, 1);
            }
            
        }
        
        Set<Integer> allClass = classsCount.keySet();
        Integer[] allClassArr = allClass.toArray(new Integer[allClass.size()]);
        
        
        
        //To find the default distribution
        //Class,Distribution
        HashMap<Integer,Double> distribution=new HashMap<Integer,Double>();
        
        for(int i=0;i<allClassArr.length;i++)
        {
            
            distribution.put(allClassArr[i],(double) (classsCount.get(allClassArr[i])/examples.size()));
            
        }
        
        ArrayList<Integer> exampleList = new ArrayList<Integer>();
        
        for(int i=0;i<examples.size();i++)
        {
            exampleList.add(i);
        }
        
        ArrayList<Integer> exampleList_test = new ArrayList<Integer>();
        
        for(int i=0;i<examples_test.size();i++)
        {
            exampleList_test.add(i);
        }
        
        //To create an attribute list
        ArrayList<Integer> attributes = new ArrayList<Integer>();
        
        
        
        for(int i=1;i<=length-1;i++)
        {
            attributes.add(i);
        }
        
        //To fill details in classDetails
        classDetails = new HashMap<Integer,Integer>();
        
        for(int i=0;i<allClassArr.length;i++)
        {
            classDetails.put(i, allClassArr[i]);
        }
        
        dtree dt = new dtree(classDetails, pruning_thr,matrix);
        
        //Optimized
        if(option.equalsIgnoreCase("optimized")){
            Tree tree = dt.oDLT(exampleList, attributes, distribution, pruning_thr);
            dt.displayTrainingResult(tree,0);
            
            dt.optimizedRanClassification(tree, exampleList_test, matrix_test,attributes);
        }
        
        //Random
        else if(option.equalsIgnoreCase("randomized")){
            Tree treern = dt.rDLT(exampleList, attributes, distribution, pruning_thr);
            dt.displayTrainingResult(treern,0);
            dt.optimizedRanClassification(treern, exampleList_test, matrix_test,attributes);
        }
        //Random 3
        else if(option.equalsIgnoreCase("forest3")){
            ArrayList<Tree> list3 = new ArrayList<Tree>();
            
            for(int i=0;i<3;i++)
            {
                list3.add(dt.rDLT(exampleList, attributes, distribution, pruning_thr));
            }
            for(int i=0;i<3;i++)
            {
                dt.displayTrainingResult(list3.get(i),i);
            }
            
            dt.RanClassification3(list3, exampleList_test, matrix_test,attributes);
        }
        //Random15
        else if(option.equalsIgnoreCase("forest15")){
            ArrayList<Tree> list = new ArrayList<Tree>();
            
            for(int i=0;i<15;i++)
            {
                list.add( dt.rDLT(exampleList, attributes, distribution, pruning_thr));
            }
            
            for(int i=0;i<15;i++)
            {
                dt.displayTrainingResult(list.get(i),i);
            }
            dt.RanClassification3(list, exampleList_test, matrix_test,attributes);
        }
        
        bufferReader.close();
        bufferReader_test.close();
    }
    
}
