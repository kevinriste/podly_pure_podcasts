export type FeedSortOption =
  | 'newest'
  | 'oldest'
  | 'title-asc'
  | 'title-desc'
  | 'feed-added-oldest'
  | 'feed-added-newest';

const FEED_LIST_SORT_STORAGE_KEY = 'podly:feed-list-sort';
const VALID_FEED_SORT_OPTIONS = new Set<FeedSortOption>([
  'newest',
  'oldest',
  'title-asc',
  'title-desc',
  'feed-added-oldest',
  'feed-added-newest',
]);

export function loadFeedListSortPreference(): FeedSortOption {
  if (typeof window === 'undefined') {
    return 'newest';
  }

  const rawValue = window.localStorage.getItem(FEED_LIST_SORT_STORAGE_KEY);
  if (rawValue === 'title') {
    return 'title-asc';
  }
  return VALID_FEED_SORT_OPTIONS.has(rawValue as FeedSortOption)
    ? (rawValue as FeedSortOption)
    : 'newest';
}

export function persistFeedListSortPreference(sortBy: FeedSortOption): void {
  if (typeof window === 'undefined') {
    return;
  }

  window.localStorage.setItem(FEED_LIST_SORT_STORAGE_KEY, sortBy);
}
