package edu.umn.cs.recsys;

import com.google.common.collect.Sets;
import edu.umn.cs.recsys.dao.ItemTagDAO;
import it.unimi.dsi.fastutil.longs.LongArraySet;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.core.LenskitRecommender;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import java.util.*;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;

/**
 * A metric that measures the tag entropy of the recommended items.
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class TagEntropyMetric extends AbstractMetric<TagEntropyMetric.Context, TagEntropyMetric.Result, TagEntropyMetric.Result> {
    private final int listSize;

    /**
     * Construct a new tag entropy metric.
     * 
     * @param nitems The number of items to request.
     */
    public TagEntropyMetric(int nitems) {
        super(Result.class, Result.class);
        listSize = nitems;
    }

    @Override
    public String getSuffix() {
        return Integer.toString(listSize);
    }

    /**
     * Make a metric accumulator.  Metrics operate with <em>contexts</em>, which are created
     * for each algorithm and data set.  The context is provided when computing each user's output,
     * and is usually used to accumulate the results into a global statistic for the whole
     * evaluation.
     *
     * @param algorithm The algorithm being tested.
     * @param data The data set being tested with.
     * @param rec The recommender instance for the current algorithm (if applicable).
     * @return An accumulator for analyzing this algorithm and data set.
     */
    @Override
    public Context createContext(Attributed algorithm, TTDataSet data, Recommender rec) {
        return new Context((LenskitRecommender) rec);
    }

    /**
     * Actually measure the performance or other value over a user's results.
     * @param user The user.
     * @param context The context or accumulator.
     * @return The results for this user, to be written to the per-user output file.
     */
    @Override
    protected Result doMeasureUser(TestUser user, Context context) {
        List<ScoredId> recommendations =
                user.getRecommendations(listSize,
                                        ItemSelectors.allItems(),
                                        ItemSelectors.trainingItems());
        if (recommendations == null) {
            return null; // no results for this user.
        }
        // get tag data from the context so we can use it
        ItemTagDAO tagDAO = context.getItemTagDAO();
        TagVocabulary vocab = context.getTagVocabulary();
        
        double entropy = 0;
        
        // TODO Implement the entropy metric
        LongSet distinctMovies = new LongArraySet();
        Set<String> seenTags = Sets.newHashSet();
        MutableSparseVector eVec = vocab.newTagVector();
        eVec.fill(0.0);

        for(ScoredId s : recommendations)
        {
            if(!distinctMovies.contains(s.getId()))
            {
                distinctMovies.add(s.getId());
            }
        }
        double recommendationSize = distinctMovies.size();

        if(recommendationSize > 0)
        {
            for(long m : distinctMovies)
            {
                List<String> itemTags = tagDAO.getItemTags(m);
                Set<String> movieTags = Sets.newHashSet(itemTags);

                double tm = movieTags.size();
                MutableSparseVector ptlStore = vocab.newTagVector();

                for(String t : movieTags)
                {
                    if(vocab.hasTag(t) || !seenTags.contains(t))
                    {
                        seenTags.add(t);
                    }

                    double ptl = (1/tm) * (1/recommendationSize);
                    double hl = -ptl * (Math.log(ptl)/ Math.log(2));

                    ptlStore.set(vocab.getTagId(t), hl);
                }
                eVec.add(ptlStore);
            }
        }
        entropy = eVec.sum();
        /*
        Map<Long, HashSet<String>> movieTags = new HashMap<Long, HashSet<String>>();
        HashSet<String> distinctTags = new HashSet<String>();
        String s1 = "";
        double sumOfL = 0;
        double sumofPx = 0;
        int recommendationSize = recommendations.size();

        try
        {
            if(recommendationSize > 0)
            {
                MutableSparseVector tagVector = vocab.newTagVector();
                for(ScoredId scoreId : recommendations)
                {
                    List<String> tags = tagDAO.getItemTags(scoreId.getId());
                    for(String s : tags)
                    {
                        if(!distinctTags.contains(s))
                        {
                            distinctTags.add(s);
                        }
                    }
                    for(String s2 : tags)
                    {
                        s1 += " ";
                        s1 += s2;
                    }

                    movieTags.put(scoreId.getId(), distinctTags);
                }

                for(Long mId : movieTags.keySet())
                {
                    sumOfL += movieTags.get(mId).size();
                }

                for(ScoredId sId: recommendations)
                {
                    HashSet<String> tags = movieTags.get(sId.getId());
                    for(String t : tags)
                    {
                        if(tagVector.containsKey(vocab.getTagId(t)) == false)
                        {

                        }
                    }
                }
            }
        }
        catch(Exception e)
        {
            System.out.println("Error : " + e.toString());
        }
        */



        // record the entropy in the context for aggregation
        context.addUser(entropy);

        // and finally return this user's evaluation results
        return new Result(entropy);
    }

    /**
     * Get the aggregate results for an experimental run.
     * @param context The context for the experimental run.
     * @return The aggregate results for the experimental run.
     */
    @Override
    protected Result getTypedResults(Context context) {
        return new Result(context.getMeanEntropy());
    }

    /**
     * Result type for the entropy metric. This encapsulates the entropy and gives it a column name.
     */
    public static class Result {
        @ResultColumn("TagEntropy")
        public final double entropy;

        public Result(double entropy) {
            this.entropy = entropy;
        }
    }

    /**
     * Context class for accumulating the total entropy across users.  This context also remembers
     * the recommender, so we can get the tag data.
     */
    public static class Context {
        private LenskitRecommender recommender;
        private double totalEntropy;
        private int userCount;

        /**
         * Create a new context for evaluating a particular recommender.
         * @param rec The recommender being evaluated.
         */
        public Context(LenskitRecommender rec) {
            recommender = rec;
        }

        /**
         * Get the recommender being evaluated.
         * @return The recommender being evaluated.
         */
        public LenskitRecommender getRecommender() {
            return recommender;
        }

        /**
         * Get the item tag DAO for this evaluation context.
         * @return A DAO providing access to the tag lists of items.
         */
        public ItemTagDAO getItemTagDAO() {
            return recommender.get(ItemTagDAO.class);
        }

        /**
         * Get the tag vocabulary for the current recommender evaluation.
         * @return The tag vocabulary for this evaluation context.
         */
        public TagVocabulary getTagVocabulary() {
            return recommender.get(TagVocabulary.class);
        }

        /**
         * Add the entropy for a user to this context.
         * @param entropy The entropy for one user.
         */
        public void addUser(double entropy) {
            totalEntropy += entropy;
            userCount += 1;
        }

        /**
         * Get the average entropy over all users.
         * @return The average entropy over all users.
         */
        public double getMeanEntropy() {
            return totalEntropy / userCount;
        }
    }
}
