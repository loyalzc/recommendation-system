# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/2 13:41
@Function:
"""
import json
import pickle as pickle

DataPath = "playlistdetail.all.json"
out_file_path1 = "all_song_list.csv"
out_file_path2 = "std_song_list.csv"


def parse_song_line(in_line):
    """
    从源数据中读取所需要的数据
    歌单：名字；标签；订阅数；id；歌曲列表
    歌曲：歌曲id；歌曲名字；演唱者；流行度
    :param in_line:
    :return:
    """
    data = json.loads(in_line)
    song_list_name = data['result']['name']
    song_list_tags = ','.join(data['result']['tags'])
    song_list_subscribed_count = data['result']['subscribedCount']

    if song_list_subscribed_count < 100:
        return False

    song_list_id = data['result']['id']
    song_info = ""
    songs = data['result']['tracks']
    song_list = ""
    if len(songs) == 0:
        return False

    for song in songs:
        try:
            if song['id'] == "":
                continue
            song_info += "\t" + ":::".join([str(song['id']), song['name'], song['artists'][0]['name'], str(song['popularity'])])
        except Exception:
            continue
    song_list = song_list_name + "##" + song_list_tags + "##" + str(song_list_id) + "##" + str(song_list_subscribed_count) + song_info
    return song_list


def parse_song_info(song_info):
    try:
        song_id, song_name, song_atrists, song_popularity = song_info.split(":::")
        # 组装成特定格式 :其中评分和 timestamp随便加入
        return ",".join([song_id, "1.0", '10000'])
    except Exception:
        return ""


def parse_song_list(song_list):
    """
    组装标准的推荐系统格式：user_id，item_id，score, timestamp
    :param song_list:
    :return:
    """
    try:
        contents = song_list.split("\t")
        song_list_name, song_list_tags, song_list_id, song_list_subscribed_count = contents[0].split("##")
        song_info = map(lambda x: song_list_id + "," + parse_song_info(x), contents[1:])
        return "\n".join(song_info)
    except Exception as e:
        print(e)
        return False


def parse_file(in_file, out_file):
    with open(out_file, 'w', encoding='utf-8') as out_f:
        with open(in_file, encoding='utf-8') as in_f:
            for line in in_f:
                # song_list = parse_song_line(line)
                song_list = parse_song_list(line)

                if song_list:
                    out_f.write(song_list + "\n")
    out_f.close()


def get_songs_info(in_line, song_list_dic, song_dic):
    """
    获取歌曲id对应的歌曲名称
    :param in_line: 每一条歌单记录
    :param song_list_dic: 歌单字典
    :param song_dic: 歌曲字典
    :return:
    """
    contents = in_line.split("\t")
    song_list_name, song_list_tags, song_list_id, song_list_subscribed_count = contents[0].split("##")
    song_list_dic[song_list_id] = song_list_name
    for song_info in contents[1:]:
        try:
            song_id, song_name, song_atrists, song_popularity = song_info.split(":::")
            song_dic[song_id] = song_name + "\t" + song_atrists
        except:
            print("song format error!")
            print(song_info)


def parse_file_info(in_file, out_song_list, out_song):

    song_list_dic = {}
    song_dic = {}
    with open(in_file, encoding='utf8') as in_f:
        for line in in_f:
            get_songs_info(line, song_list_dic, song_dic)
    pickle.dump(song_list_dic, open(out_song_list, "wb"))
    # 可以通过 playlist_dic = pickle.load(open("playlist.pkl","rb"))重新载入
    pickle.dump(song_dic, open(out_song, "wb"))


if __name__ == '__main__':
    # parse_file(DataPath, out_file_path1)
    parse_file(out_file_path1, out_file_path2)
    # parse_file_info(out_file_path1, "song_list_dic", "song_dic")