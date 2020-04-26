import pandas as pd
import numpy as np
import torch


class Dataset(object):
    def __init__(self, path, sep=',', session_key='sessionid', item_key='itemid', time_key='timestamp',
                 n_sample=-1, itemmap=None, itemstamp=None, time_sort=False):
        """
        Args:
            path: path of the csv file
            sep: separator for the csv
            session_key, item_key, time_key: name of the fields corresponding to the sessions, items, time
            n_samples: the number of samples to use. If -1, use the whole dataset.
            itemmap: mapping between item IDs and item indices
            time_sort: whether to sort the sessions by time or not
        """
        # Read event stream and remove useless columns (just as of now)
        try:
            self.df = pd.read_csv(path, sep=sep,
                                      dtype={session_key: int, item_key: int, time_key: float},
                                      error_bad_lines=False)
        except:
            self.df = pd.read_pickle(path)
        # 우선은 삭제
        # self.df.drop(columns=['event', 'visitorid', 'transactionid', 'datetime', 'timedelta', 'timedelta_datetime'],
        #              inplace=True)

        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort
        if n_sample > 0:
            self.df = self.df[:n_sample]

        # Add colummn item index to data
        self.add_item_indices(itemmap=itemmap)
        """
        Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        clicks within a session are next to each other, where the clicks within a session are time-ordered.
        """
        self.df.sort_values([session_key, time_key], inplace=True)
        self.click_offsets = self.get_click_offset()
        self.session_idx_arr = self.order_session_idx()

    def add_item_indices(self, itemmap=None):
        """
        Add item index column named "item_idx" to the df
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # type is numpy.ndarray
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            # Build itemmap is a DataFrame that have 2 columns (self.item_key, 'item_idx)
            itemmap = pd.DataFrame({self.item_key: item_ids,
                                   'item_idx': item2idx[item_ids].values})
        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')

    def get_item_index_dict(self):
        item_ids = self.df[self.item_key].unique()
        idx_to_item = pd.Series(item_ids).to_dict()
        item_to_idx = {y: x for x, y in idx_to_item.items()}
        return item_to_idx, idx_to_item

    def get_click_offset(self):
        """
        해당 Session이 첫 Session에서 몇번째 해당하는 것인지
         - Return the offsets of the beginning clicks of each session IDs,
           where the offset is calculated against the first click of the first session ID.

        self.df[self.session_key] return a set of session_key
        self.df[self.session_key].nunique() return the size of session_key set (int)
        self.df.groupby(self.session_key).size() return the size of each session_id
        self.df.groupby(self.session_key).size().cumsum() retunn cumulative sum
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets

    def order_session_idx(self):
        """
        Returns: Session Index Array

        """
        if self.time_sort:
            # starting time for each sessions, sorted by session IDs
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            # order the session indices by session starting times
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr

    @property
    def items(self):
        return self.itemmap[self.item_key].unique()


class DataLoader():
    def __init__(self, dataset, batch_size=50):
        """
        A class for creating session-parallel mini-batches.

        Args:
             dataset (SessionDataset): the session dataset to generate the batches from
             batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.

        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """
        # initializations
        df = self.dataset.df
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]       #Session에서 시작 위치
        end = click_offsets[session_idx_arr[iters] + 1]     #Session에서 종 위치
        mask = []  # indicator for the sessions to be terminated
        finished = False

        # COUNTER = 0
        while not finished:
            # COUNTER += 1
            # print(f'Current Loop Count : {COUNTER}')

            minlen = (end - start).min()                    #Session길이 중에 최소값
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]          #Session 시작에서의 item index

            # 최소 Session 길이만큼만 Session Sequence를 찾아내서 idx_input(start), idx_targt(start+1)의 값이 된다.
            # 예를 들어, 최소 Session 길이가 4이고, 내부 Item Sequence가 [0,1,2,3]라면, input-target 은 [0-1, 1-2, 2-3]이 된다.
            # 그리고, batch_size만큼 Vector로 만들게 된다.
            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target                          #Session 시작에서의 item index
                idx_target = df.item_idx.values[start + i + 1]  #Session 시작에서 i+1 이후의 item index
                input = torch.LongTensor(idx_input)             #Shape = (batch_size,)
                target = torch.LongTensor(idx_target)           #Shape = (batch_size,)
                yield input, target, mask

            # click indices where a particular session meets second-to-last element
            # Start를 한칸 이동하고, 얼마나 많은 Session이 끝나는지 아래에서 확인
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            # end-start <= 1이면, 이제 Session이 0또는 1개밖에 안남음
            # 즉, mask = 종료해야할 Sesession Index
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
