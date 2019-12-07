import numpy



from nltk.tokenize import TweetTokenizer
from nltk import tokenize


class Features():


    def __init__(self):
        self.tokenizer = TweetTokenizer()

    
    def thread_info(self, thread):
        output = []
        unique_reply_dict = {}
        for thread_post in thread['posts']:
            if 'in_reply_to' in thread_post:
                if thread_post['in_reply_to'] in unique_reply_dict:
                    unique_reply_dict[thread_post['in_reply_to']] += 1
                else:
                    unique_reply_dict[thread_post['in_reply_to']] = 1
        branch_num = len(unique_reply_dict)
        is_self_post = 1 if 'is_self_post' in thread and thread['is_self_post'] else 0
        # Total Number of posts
        output.append(numpy.full(300, 1.0 * len(thread['posts'])))
        # Number of unique branches
        output.append(numpy.full(300, 1.0 * branch_num))
        # Average Length of branches
        output.append(numpy.full(300, 1.0 * self.average_branches(unique_reply_dict, branch_num)))
        # Whether it is a self post
        output.append(numpy.full(300, 1.0 * is_self_post))
        return output

    def average_branches(self, branch_obj, branch_num):
        sum_of_branch = sum([branch_obj[key] for key in branch_obj])
        return sum_of_branch / branch_num

    def isSameAuthor(self, thread, post):
        try:
            reply_id = post['in_reply_to']
            if self.getAuthorFromID(thread, reply_id) == post['author']:
                return numpy.full(300, 1.0)
            else:
                return numpy.full(300, 0.0)
        except:
            return numpy.full(300, 0.0)

    def getAuthorFromID(self, thread, target_id):
        for post in thread['posts']:
            if post['id'] == target_id:
                try:
                    return post['author']
                except:
                    return None

    # returns body given the post id
    def getBodyFromID(self, thread, target_id):
        for post in thread['posts']:
            if post['id'] == target_id:
                return post['body']


    # get the depth of the comment
    def getStructureFeatures(self, thread, target_id):
        output = []
        p_output = [numpy.zeros(300), numpy.zeros(300), numpy.zeros(300)]

        post_iter = None
        for post in thread['posts']:
            if post['id'] == target_id:
                # get the depth
                output.append(self.getDepth(post, 300))
                # get the character count
                output.append(numpy.full(300, 1.0 * len(post['body'])))
                # get the word count
                output.append(numpy.full(300, 1.0 * len(self.tokenizer.tokenize(post['body']))))
                # get the sentence count
                output.append(numpy.full(300, 1.0 * len(tokenize.sent_tokenize(post['body']))))

                post_iter = post
                break
        

        if 'in_reply_to' in post_iter:
            parent_id = post_iter['in_reply_to']
            for p in thread['posts']:
                if p['id'] == parent_id:
                    p_output[0] += numpy.full(300, 1.0 * len(p['body']))
                    # get the word count
                    p_output[1] += numpy.full(300, 1.0 * len(self.tokenizer.tokenize(p['body'])))
                    # get the sentence count
                    p_output[2] += numpy.full(300, 1.0 * len(tokenize.sent_tokenize(p['body'])))
                    break

        # parent = False
        # found = False
        
        # while not parent:
        #     parent_id = post_iter['in_reply_to']
        #     for p in thread['posts']:
        #         if p['id'] == parent_id:
        #             post_iter = p
        #             found = True
        #             p_output[0] += numpy.full(300, 1.0 * len(p['body']))
        #             # get the word count
        #             p_output[1] += numpy.full(300, 1.0 * len(self.tokenizer.tokenize(p['body'])))
        #             # get the sentence count
        #             p_output[2] += numpy.full(300, 1.0 * len(tokenize.sent_tokenize(p['body'])))
            
        #     if not found:
        #         break
        #     else:
        #         found = False
        #     if 'in_reply_to' not in post_iter:
        #         parent = True
    
        return output + p_output

    def getDepth(self, post, length):
        try:
            return numpy.full(length, 1.0 * post['post_depth'])
        except:
            return numpy.zeros(length)

    def getParentBody(self, thread, target_id):
        for post in thread['posts']:
            if post['id'] == target_id:
                try:
                    return post['body']
                except:
                    return ""
        
        return ""
    

    def test(self):
        print("features setup correctly")


