from naver_movie_scraper import scrap_comments
import idx_parser

for page_numbers in range(24,26,1):
    a_tag_href_attribute_list = idx_parser.make_a_tag_href_attribute_list(page_numbers)
    idx_list = idx_parser.extract_idx_from_a_tag_href_attribute_list(a_tag_href_attribute_list)


    total_pos_tag_count = 0
    total_nag_tag_count = 0

    for idx in idx_list:
    # pos_tag_count = 0
    # nag_tag_count = 0
        scrap_list_of_dict = scrap_comments(idx, limit=100)
        with open('./result/'+'page_'+str(page_numbers)+'/'+str(idx)+'_result.txt','w',encoding='utf-8') as f:
            for dict in scrap_list_of_dict:
                user_id = dict['user']
                score = int(dict['score'])
                text = dict['text']
                if score > 8:
                    pos_or_nag_tag = "1"
                    total_pos_tag_count = total_pos_tag_count + 1
                    f.write('\\t'.join([user_id,text,pos_or_nag_tag])+'\n')
                elif score < 5:
                    pos_or_nag_tag = "0"
                    total_nag_tag_count = total_nag_tag_count + 1
                    f.write('\\t'.join([user_id,text,pos_or_nag_tag])+'\n')
                else:
                    continue

    with open('./result/' + 'page_' + str(page_numbers) + '/' + str(
            page_numbers) + '_pos_' + str(total_pos_tag_count) + '_nag_' + str(total_nag_tag_count) + '.txt', 'w',
              encoding='utf-8') as f:
        print("end")
